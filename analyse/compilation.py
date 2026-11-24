"""Join detected crossing intervals in event CSV order."""

import csv
import logging
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from .common import DetectorName


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
    def compile_output(cls, events_csv: Path):
        with events_csv.open(encoding="utf-8", newline="") as table:
            intervals = [(Path(row["file_path"]), float(row["start_s"]), float(row["end_s"]))
                         for row in csv.DictReader(table) if row["event"] == DetectorName.SPEEDBREAKER]
        path = events_csv.with_suffix(".speedbreakers.mp4")
        if not intervals:
            path.unlink(missing_ok=True)
            logger.info("Skipping compilation because %s contains no speedbreaker events", events_csv)
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
        logger.info("Saved speedbreaker compilation to %s", path)
