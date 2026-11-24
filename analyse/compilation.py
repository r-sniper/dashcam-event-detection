"""Join detected crossing intervals in event CSV order."""

import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory


logger = logging.getLogger(__name__)


class SpeedbreakerCompilation:
    @staticmethod
    def write_segment(source: Path, start: float, end: float, target: Path):
        # FFmpeg selects the audio stream with the most channels, preferring the first
        # on ties. The mono silence input supplies audio for clips recorded on mute.
        subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-ss", str(start), "-i", str(source),
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", f"{end - start:.3f}", "-sn",
            "-vf", "fps=30,scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                   "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-af", "apad", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(target),
        ], check=True)

    @classmethod
    def write(cls, path: Path, intervals: Sequence[tuple[Path, float, float]]):
        if not intervals:
            path.unlink(missing_ok=True)
            return

        logger.info("Compiling %d speedbreaker clips to %s", len(intervals), path)
        with TemporaryDirectory(dir=path.parent) as directory:
            folder = Path(directory)
            manifest = folder / "clips.ffconcat"
            with manifest.open("w", encoding="utf-8") as clips:
                for i, (source, start, end) in enumerate(intervals):
                    segment = folder / f"{i}.mov"
                    cls.write_segment(source, start, end, segment)
                    clips.write(f"file {segment.name}\n")
            video = folder / "compilation.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "concat", "-i", manifest.name,
                            "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", video.name],
                           cwd=folder, check=True)
            video.replace(path)
