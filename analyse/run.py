import argparse
import csv
import os
import tempfile
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import TypedDict, cast

import cv2

from .common import Detection, DetectorName, FrontClips
from .preprocess import DEFAULT_WORK_DIR, VideoPreprocessor
from .signals import MotionCache
from .speedbreaker import SpeedbreakerDetector
from .warning_sign import WarningSignDetector


class EncounterField(TypedDict, total=False):
    encounter_id: int


class EventRow(EncounterField):
    """Common fields of a recorded event, before detector-specific columns."""

    filename: str
    file_path: str
    event: str
    peak_s: float
    score: float


class ClipRow(TypedDict):
    filename: str
    file_path: str
    detector: str
    scan_mode: str
    duration_s: float
    inspected_s: float
    events: int


ScanResult = tuple[list[EventRow], ClipRow]
Detector = SpeedbreakerDetector | WarningSignDetector


class AnalysisRun:
    detectors: dict[DetectorName, type[Detector]] = {
        SpeedbreakerDetector.name: SpeedbreakerDetector,
        WarningSignDetector.name: WarningSignDetector,
    }
    event_fields = ["encounter_id", "filename", "file_path", *dict.fromkeys(
        field.name for detector in detectors.values() for field in fields(detector.result_type)
    )]
    clip_fields = ["filename", "file_path", "detector", "scan_mode", "duration_s", "inspected_s", "events"]

    def __init__(self, names: Sequence[DetectorName], scan_mode: str = "full", cache: MotionCache | None = None,
                 hwaccel: str | None = None, sign_lookback: float | None = None,
                 work_dir: Path = DEFAULT_WORK_DIR, preprocess_cache: bool = False):
        self.scan_mode = scan_mode
        self.sign_lookback = sign_lookback
        self.preprocessor = VideoPreprocessor(work_dir, preprocess_cache, include_motion=DetectorName.SPEEDBREAKER in names,
                                              motion_cache=cache, hwaccel=hwaccel)
        if sign_lookback is not None:
            names = sorted(names, key=lambda name: name != DetectorName.SPEEDBREAKER)
        self.selected: list[Detector] = []
        for name in names:
            detector_type = self.detectors[name]
            if issubclass(detector_type, SpeedbreakerDetector):
                self.selected.append(detector_type(scan_mode))
            else:
                self.selected.append(detector_type())

    @staticmethod
    def record(video: Path, detector: DetectorName, mode: str, duration: float, inspected: float,
               detections: Iterable[Detection]) -> ScanResult:
        file_path = str(video.resolve())
        events = [cast(EventRow, dict(filename=video.name, file_path=file_path, **asdict(item)))
                  for item in detections]
        clip = ClipRow(filename=video.name, file_path=file_path, detector=str(detector), scan_mode=mode,
                       duration_s=round(duration, 3), inspected_s=round(inspected, 3), events=len(events))
        return events, clip

    def scan(self, video: Path) -> list[ScanResult]:
        processed = self.preprocessor.process(video)
        duration = processed.duration
        results = []
        crossings = []
        for detector in self.selected:
            mode = self.scan_mode if detector.name == DetectorName.SPEEDBREAKER else "full"
            if isinstance(detector, WarningSignDetector) and self.sign_lookback is not None:
                windows = detector.approaches(crossings, duration, self.sign_lookback)
                events, inspected = detector.detect(processed, windows)
                mode = f"approach-{self.sign_lookback:g}s"
            else:
                events, inspected = detector.detect(processed)
            if detector.name == DetectorName.SPEEDBREAKER:
                crossings = events
            results.append(self.record(video, detector.name, mode, duration, inspected, events))
        return results

    @staticmethod
    def assign_ids(events: Sequence[EventRow], clips: Sequence[ClipRow]):
        durations = {row["file_path"]: row["duration_s"] for row in clips}
        previous = None
        next_id = 0
        for event in sorted(events, key=lambda e: (e["file_path"], e["peak_s"], e["event"])):
            shared_id = None
            if event["event"] == DetectorName.SPEEDBREAKER and previous is not None:
                prev_path, path = Path(previous["file_path"]), Path(event["file_path"])
                if (prev_path != path and prev_path.parent == path.parent and event["peak_s"] <= 2.5
                        and durations[str(prev_path)] - previous["peak_s"] <= 2.5):
                    try:
                        prev_start, start = [datetime.strptime(p.stem[:16], "%Y_%m%d_%H%M%S")
                                             for p in (prev_path, path)]
                    except ValueError:
                        pass
                    else:
                        seam_gap = (start - prev_start).total_seconds() - durations[str(prev_path)]
                        gap = (start - prev_start).total_seconds() + event["peak_s"] - previous["peak_s"]
                        if abs(seam_gap) <= 1 and 0 <= gap <= 2.5:
                            shared_id = previous["encounter_id"]
            if shared_id is not None:
                event["encounter_id"] = shared_id
            else:
                next_id += 1
                event["encounter_id"] = next_id
            if event["event"] == DetectorName.SPEEDBREAKER:
                previous = event

    @staticmethod
    def write_table(path: Path, fields: Collection[str], rows: Iterable[Mapping[str, object]]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", newline="",
                                         suffix=".tmp", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(f.name, path)

    @classmethod
    def save(cls, results: Iterable[list[ScanResult]], output: Path) -> tuple[list[EventRow], list[ClipRow]]:
        events, clips = [], []
        for per_video in results:
            for found, clip in per_video:
                events.extend(found)
                clips.append(clip)
        cls.assign_ids(events, clips)
        events.sort(key=lambda e: (e["file_path"], e["peak_s"], e["event"]))
        cls.write_table(output, cls.event_fields, events)
        cls.write_table(output.with_suffix(".clips.csv"), cls.clip_fields, clips)
        return events, clips


def main():
    p = argparse.ArgumentParser(description="Find events in a list or folder of Viofo front-camera MP4s.")
    p.add_argument("inputs", type=Path, nargs="+", help="Front MP4s, or folders containing them")
    available = ", ".join(AnalysisRun.detectors)
    p.add_argument("--detect", default=DetectorName.SPEEDBREAKER.value, help=f"Comma-separated: {available}")
    # Accept the explicit full mode used by earlier CLI commands.
    p.add_argument("--scan-mode", choices=("full",), help=argparse.SUPPRESS)
    p.add_argument("--output", type=Path, default=Path("analysis/speedbreaker/detections.csv"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
                   help="Store preprocessed clips here (default: %(default)s)")
    p.add_argument("--preprocess-cache", action="store_true",
                   help="Reuse matching preprocessed clips. By default, extract and save them again.")
    p.add_argument("--motion-cache", type=Path, help="Reuse full-video motion measurements in this folder")
    p.add_argument("--hwaccel", choices=["d3d11va"], help="Use Windows GPU decoding for 8-bit motion analysis")
    p.add_argument("--sign-lookback", type=float, help="Search for signs this many seconds before each detected crossing")
    p.add_argument("--review", type=Path, help="Write an offline HTML video review page")
    args = p.parse_args()
    requested = list(dict.fromkeys(n.strip() for n in args.detect.split(",")))
    unknown = set(requested) - AnalysisRun.detectors.keys()
    if unknown:
        p.error(f"Unknown detector(s): {', '.join(sorted(unknown))}. Available: {available}")
    names = [DetectorName(name) for name in requested]
    if args.sign_lookback is not None:
        required = {DetectorName.SPEEDBREAKER, DetectorName.WARNING_SIGN}
        if args.sign_lookback <= 0 or not required.issubset(names):
            p.error("--sign-lookback requires a positive duration and both speedbreaker and warning_sign")
    if args.output.suffix.lower() != ".csv" or args.workers < 1:
        p.error("--output needs a .csv filename and --workers must be positive")
    cv2.setNumThreads(1)
    cache = MotionCache(args.motion_cache) if args.motion_cache else None
    run = AnalysisRun(names, cache=cache, hwaccel=args.hwaccel, sign_lookback=args.sign_lookback,
                      work_dir=args.work_dir, preprocess_cache=args.preprocess_cache)
    try:
        videos = FrontClips.resolve(args.inputs)
    except ValueError as err:
        p.error(str(err))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        def results() -> Iterator[list[ScanResult]]:
            for i, per_video in enumerate(pool.map(run.scan, videos), 1):
                print(f"[{i}/{len(videos)}] {videos[i-1].name}", flush=True)
                yield per_video
        events, clips = run.save(results(), args.output)
    if args.review:
        from visualize.review import ReviewPage
        ReviewPage.write(args.review, events, clips)
    print(f"{len(events)} detections across {len(videos)} clips. Saved {args.output}")
