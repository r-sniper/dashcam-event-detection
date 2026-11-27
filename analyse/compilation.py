"""Render crossing clips and join them in event order."""

import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from .preprocess import FFprobeResult


logger = logging.getLogger(__name__)


@dataclass
class SourceStreams:
    video: dict[str, Any]
    audio: dict[str, Any] | None

    @classmethod
    def read(cls, source: Path) -> "SourceStreams":
        streams = cast(list[dict[str, Any]], FFprobeResult.read(source).raw["streams"])
        video = next(s for s in streams if s["codec_type"] == "video")
        audio = next((s for s in streams if s["codec_type"] == "audio"), None)
        return cls(video, audio)


class SpeedbreakerCompilation:
    @staticmethod
    def write_clip(source: Path, start: float, end: float, target: Path):
        streams = SourceStreams.read(source)
        codec = streams.video["codec_name"]
        command = ["ffmpeg", "-v", "error", "-nostdin", "-ss", str(start), "-i", str(source),
                   "-t", f"{end - start:.3f}", "-map", "0:v:0", "-c:v", codec,
                   "-fps_mode:v", "passthrough", "-enc_time_base:v", streams.video["time_base"]]
        if codec in ("h264", "hevc"):
            command += ["-preset", "veryfast", "-tag:v", streams.video["codec_tag_string"]]
        if "bit_rate" in streams.video:
            command += ["-b:v", streams.video["bit_rate"]]
        if streams.audio:
            command += ["-map", "0:a:0", "-c:a", streams.audio["codec_name"], "-af", "apad"]
            if "bit_rate" in streams.audio:
                command += ["-b:a", streams.audio["bit_rate"]]
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=target.parent) as directory:
            clip = Path(directory) / "clip.mp4"
            subprocess.run([*command, "-movflags", "+faststart", str(clip)], check=True)
            clip.replace(target)

    @staticmethod
    def prepare_segment(source: Path, target: Path, streams: SourceStreams,
                        audio: dict[str, Any] | None, timescale: str):
        # Copy the clipped video. Give concat a common time base and continuous, uncompressed audio.
        command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(source)]
        audio_options = []
        if audio is not None:
            rate = audio["sample_rate"]
            layout = audio.get("channel_layout") or f"{audio['channels']}c"
            if streams.audio is None:
                command += ["-f", "lavfi", "-i", f"anullsrc=r={rate}:cl={layout}"]
            audio_options = ["-map", "0:a:0" if streams.audio else "1:a:0", "-af", "apad",
                             "-ar", rate, "-channel_layout", layout, "-c:a", "pcm_f32le"]
        subprocess.run([*command, "-t", streams.video["duration"], "-map", "0:v:0", "-c:v", "copy",
                        "-video_track_timescale", timescale, *audio_options, str(target)], check=True)

    @classmethod
    def join(cls, clips: Sequence[Path], path: Path):
        logger.info("Compiling %d speedbreaker clips to %s", len(clips), path)
        sources = {clip: SourceStreams.read(clip) for clip in clips}
        video_format = next(iter(sources.values())).video
        for source, streams in sources.items():
            if any(streams.video[k] != video_format[k] for k in ("codec_name", "width", "height", "pix_fmt")):
                raise ValueError(f"{source}: compilation clips must share a video codec, resolution and pixel format")
        # The concat demuxer needs one timestamp scale even when frame rates differ.
        timescale = video_format["time_base"].split("/")[1]
        audio = next((s.audio for s in sources.values() if s.audio is not None), None)
        with TemporaryDirectory(dir=path.parent) as directory:
            folder = Path(directory)
            manifest = folder / "clips.ffconcat"
            with manifest.open("w", encoding="utf-8") as entries:
                for i, source in enumerate(clips):
                    segment = folder / f"{i}.mov"
                    cls.prepare_segment(source, segment, sources[source], audio, timescale)
                    entries.write(f"file {segment.name}\n")
            video = folder / "compilation.mp4"
            audio_options = []
            if audio is not None:
                audio_options = ["-c:a", audio["codec_name"]]
                if "bit_rate" in audio:
                    audio_options += ["-b:a", audio["bit_rate"]]
            subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "concat", "-i", manifest.name,
                            "-c:v", "copy", *audio_options, "-movflags", "+faststart", video.name],
                           cwd=folder, check=True)
            video.replace(path)
        logger.info("Saved speedbreaker compilation to %s", path)
