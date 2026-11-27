from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import NotRequired, TypedDict, cast


class DetectorName(str, Enum):
    SPEEDBREAKER = "speedbreaker"
    WARNING_SIGN = "warning_sign"

    def __str__(self) -> str:
        return self.value


@dataclass
class Detection:
    event: str
    peak_s: float
    score: float
    # Detector.detect attaches the source to each event returned by the measurement algorithm.
    file_path: Path = field(init=False, repr=False, compare=False)

    def as_row(self) -> "EventRow":
        return cast(EventRow, {**asdict(self), "filename": self.file_path.name, "file_path": str(self.file_path)})


class EventRow(TypedDict):
    encounter_id: NotRequired[int]
    filename: str
    file_path: str
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
