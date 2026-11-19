from dataclasses import dataclass
from pathlib import Path


@dataclass
class Detection:
    event: str
    peak_s: float
    score: float
    bounce_px: float | None = None
    speed_kmh: float | None = None
    # Normalized image coordinates, populated for warning signs.
    box: str = ""


class FrontClips:
    @staticmethod
    def resolve(inputs):
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
