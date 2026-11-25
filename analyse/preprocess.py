"""Prepare the shared inputs for a clip before running its detectors."""

import csv
import hashlib
import io
import json
import subprocess
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from telemetry_time import IST, parse_timestamp

from .signals import GpsTrace


DEFAULT_WORK_DIR = Path(".speedbreaker-work")


@dataclass
class GpsSample:
    timestamp_utc: str
    seconds: float | None
    latitude_deg: float | None
    longitude_deg: float | None
    speed_kmh: float | None
    heading_deg: float | None

    @staticmethod
    def number(value: str | None) -> float | None:
        try:
            number = float(value) if value else None
        except ValueError:
            return None
        return number if number is not None and np.isfinite(number) else None

    @classmethod
    def from_csv(cls, content: str, video: Path) -> list["GpsSample"]:
        try:
            start = datetime.strptime(video.stem[:16], "%Y_%m%d_%H%M%S").replace(tzinfo=IST)
        except ValueError:
            start = None
        samples = []
        for row in csv.DictReader(io.StringIO(content)):
            try:
                timestamp = parse_timestamp(row["gps_timestamp_utc"])
            except ValueError:
                continue
            offset = (timestamp - start).total_seconds() if start else None
            samples.append(cls(timestamp.isoformat(), offset, cls.number(row.get("latitude_deg")),
                               cls.number(row.get("longitude_deg")), cls.number(row.get("speed_kmh")),
                               cls.number(row.get("heading_deg"))))
        return samples


class VideoFrame(NamedTuple):
    """A sampled frame at a time in seconds from the start of the source clip."""

    timestamp_s: float
    bgr: NDArray[np.uint8]


@dataclass
class FFprobeResult:
    duration: float
    has_video_stream: bool
    raw: dict[str, object]

    @classmethod
    def read(cls, video: Path) -> "FFprobeResult":
        output = subprocess.check_output(["ffprobe", "-v", "error", "-show_format", "-show_streams",
                                          "-of", "json", str(video)], encoding="utf-8")
        raw = json.loads(output)
        duration = raw.get("format", {}).get("duration")
        if duration is None or duration == "N/A":
            raise ValueError(f"{video.name}: FFprobe did not report a duration")
        has_video_stream = any(stream.get("codec_type") == "video" for stream in raw.get("streams", []))
        return cls(duration=float(duration), has_video_stream=has_video_stream, raw=raw)


@dataclass
class ProcessedVideo:
    path: Path
    duration: float  # Seconds.
    has_video_stream: bool
    gps_samples: list[GpsSample]
    debug_metadata: dict[str, object] | None = None

    @property
    def gps_trace(self) -> GpsTrace:
        readings = {s.seconds: s.speed_kmh for s in self.gps_samples
                    if s.seconds is not None and s.speed_kmh is not None}
        seconds = sorted(readings)
        return GpsTrace(np.array(seconds), np.array([readings[t] for t in seconds]))

    def frames(self, start: float, end: float, width: int, height: int,
               fps: int) -> Iterator[VideoFrame]:
        """Stream sampled BGR frames for the requested detection window."""
        command = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "3", "-ss", str(start),
                   "-i", str(self.path), "-t", str(end-start), "-an",
                   "-vf", f"fps={fps}:round=up,scale={width}:{height}:flags=fast_bilinear",
                   "-filter_threads", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
        with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
            with cast(BinaryIO, process.stdout) as stream:
                index = 0
                while data := stream.read(width * height * 3):
                    pixels: NDArray[np.uint8] = np.ndarray((height, width, 3), dtype=np.uint8, buffer=data)
                    yield VideoFrame(timestamp_s=start + index / fps, bgr=pixels)
                    index += 1
            if process.wait():
                raise RuntimeError(f"FFmpeg could not read frames from {self.path}")

    def save(self, target: Path):
        header = json.dumps({"path": str(self.path), "duration": self.duration,
                             "has_video_stream": self.has_video_stream, "debug_metadata": self.debug_metadata,
                             "gps_samples": [asdict(sample) for sample in self.gps_samples]})
        output = NamedTemporaryFile(dir=target.parent, delete=False)
        temp = Path(output.name)
        try:
            with output:
                np.savez_compressed(output, header=header)
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)

    @classmethod
    def load(cls, target: Path) -> "ProcessedVideo":
        with np.load(target, allow_pickle=False) as data:
            header = json.loads(data["header"].item())
            return cls(path=Path(header["path"]), duration=header["duration"],
                       has_video_stream=header["has_video_stream"], debug_metadata=header["debug_metadata"],
                       gps_samples=[GpsSample(**row) for row in header["gps_samples"]])


class VideoPreprocessor:
    def __init__(self, work_dir: Path = DEFAULT_WORK_DIR, reuse_cache: bool = False):
        self.folder = work_dir / "preprocessed"
        self.reuse_cache = reuse_cache

    def process(self, video: Path) -> ProcessedVideo:
        video = video.resolve()
        stat = video.stat()
        identity = (str(video), stat.st_size, stat.st_mtime_ns)
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
        target = self.folder / f"{video.stem}.{key}.npz"
        if self.reuse_cache and target.is_file():
            return ProcessedVideo.load(target)

        probe = FFprobeResult.read(video)
        template = Path(__file__).resolve().parents[1] / "telemetry.fmt"
        telemetry = subprocess.check_output(["exiftool", "-ee3", "-n", "-f", "-p", str(template), str(video)],
                                           encoding="utf-8")
        samples = GpsSample.from_csv(telemetry, video)
        processed = ProcessedVideo(path=video, duration=probe.duration, has_video_stream=probe.has_video_stream,
                                   gps_samples=samples, debug_metadata=probe.raw)
        self.folder.mkdir(parents=True, exist_ok=True)
        processed.save(target)
        return processed
