"""The scan and output contract for a road-event detector."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .common import Detection, DetectorName, EventRow
from .preprocess import ProcessedVideo
from .signals import MotionCache


@dataclass(frozen=True)
class DetectorOptions:
    output: Path = Path("analysis/speedbreaker/detections.csv")
    scan_mode: str = "full"
    sign_lookback: float | None = None
    motion_cache: MotionCache | None = None
    hwaccel: str | None = None


type PriorDetections = Mapping[DetectorName, Sequence[Detection]]


class Detector[T: Detection](ABC):
    name: DetectorName
    result_type: type[T]
    scan_mode = "full"
    dependencies: tuple[DetectorName, ...] = ()

    def __init__(self, options: DetectorOptions = DetectorOptions()):
        self.options = options

    @abstractmethod
    def _detect(self, video: ProcessedVideo, previous: PriorDetections) -> tuple[list[T], float]:
        """Return the events and the number of seconds inspected."""

    def detect(self, video: ProcessedVideo, previous: PriorDetections = MappingProxyType({})) -> tuple[list[T], float]:
        events, inspected = self._detect(video, previous)
        source = video.path.resolve()
        for event in events:
            event.file_path = source
        return events, inspected

    @staticmethod
    def continues_encounter(previous: EventRow, current: EventRow, previous_duration: float) -> bool:
        return False

    def event_output(self, event: T):
        """Write any extra output for this event."""

    def final_output(self, events: list[T]):
        """Write any extra output for the ordered events from this run."""

    def clear_output(self):
        """Remove this detector's previous artifacts when replacing a run that used it."""
