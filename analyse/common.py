from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DetectorName(str, Enum):
    SPEEDBREAKER = "speedbreaker"
    WARNING_SIGN = "warning_sign"
    VEHICLE = "vehicle"

    def __str__(self) -> str:
        return self.value


@dataclass
class Detection:
    event: str
    peak_s: float
    score: float


class FrontClips:
    @staticmethod
    def resolve(inputs: Iterable[str | Path]) -> list[Path]:
        clips = {}
        for item in inputs:
            path = Path(item)
            candidates = sorted(path.iterdir()) if path.is_dir() else [path]
            for file in candidates:
                if file.suffix.lower() == ".mp4" and file.stem.upper().endswith("F") and file.is_file():
                    clips[file.resolve()] = None
                elif not path.is_dir():
                    raise ValueError(f"Expected an existing front-camera *F.MP4 file: {file}")
        if not clips:
            raise ValueError("No front-camera *F.MP4 files found.")
        return list(clips)
