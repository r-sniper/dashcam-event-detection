import argparse
import csv
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Collection, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import TypedDict, cast

import cv2
from tqdm.contrib.concurrent import thread_map

from .common import Detection, DetectorName, FrontClips
from .compilation import SpeedbreakerCompilation
from .preprocess import DEFAULT_WORK_DIR, ProcessedVideo, VideoPreprocessor
from .signals import MotionCache
from .speedbreaker import SpeedbreakerDetector
from .warning_sign import WarningSignDetector


logger = logging.getLogger(__name__)


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

    def __init__(self, names: Sequence[DetectorName], scan_mode: str = "full", sign_lookback: float | None = None,
                 *, motion_cache: MotionCache | None = None, hwaccel: str | None = None):
        self.scan_mode = scan_mode
        self.sign_lookback = sign_lookback
        if sign_lookback is not None:
            names = sorted(names, key=lambda name: name != DetectorName.SPEEDBREAKER)
        self.selected: list[Detector] = []
        for name in names:
            detector_type = self.detectors[name]
            if issubclass(detector_type, SpeedbreakerDetector):
                self.selected.append(detector_type(scan_mode, motion_cache=motion_cache, hwaccel=hwaccel))
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

    def run_detectors(self, video: ProcessedVideo) -> list[ScanResult]:
        results = []
        crossings = []
        for detector in self.selected:
            mode = self.scan_mode if detector.name == DetectorName.SPEEDBREAKER else "full"
            if isinstance(detector, WarningSignDetector) and self.sign_lookback is not None:
                windows = detector.approaches(crossings, video.duration, self.sign_lookback)
                events, inspected = detector.detect(video, windows)
                mode = f"approach-{self.sign_lookback:g}s"
            else:
                events, inspected = detector.detect(video)
            if detector.name == DetectorName.SPEEDBREAKER:
                crossings = events
            results.append(self.record(video.path, detector.name, mode, video.duration, inspected, events))
        return results

    def scan_videos(self, videos: Sequence[ProcessedVideo], pool: ThreadPoolExecutor) -> list[list[ScanResult]]:
        results = []
        for i, per_video in enumerate(pool.map(self.run_detectors, videos), 1):
            logger.info("[%d/%d] Scanned %s", i, len(videos), videos[i - 1].path.name)
            results.append(per_video)
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
    def collect_results(cls, results: Iterable[list[ScanResult]]) -> tuple[list[EventRow], list[ClipRow]]:
        events, clips = [], []
        for per_video in results:
            for found, clip in per_video:
                events.extend(found)
                clips.append(clip)
        cls.assign_ids(events, clips)
        events.sort(key=lambda e: (e["file_path"], e["peak_s"], e["event"]))
        return events, clips

    @classmethod
    def save(cls, events: Sequence[EventRow], clips: Sequence[ClipRow], output: Path):
        cls.write_table(output, cls.event_fields, events)
        cls.write_table(output.with_suffix(".clips.csv"), cls.clip_fields, clips)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Find events in a list or folder of Viofo front-camera MP4s.")
    p.add_argument("inputs", type=Path, nargs="+", help="Front MP4s, or folders containing them")
    available = ", ".join(AnalysisRun.detectors)
    p.add_argument("--detect", default=DetectorName.SPEEDBREAKER.value, help=f"Comma-separated: {available}")
    # Accept the explicit full mode used by earlier CLI commands.
    p.add_argument("--scan-mode", choices=("full",), help=argparse.SUPPRESS)
    p.add_argument("--output", type=Path, default=Path("analysis/speedbreaker/detections.csv"))
    p.add_argument("--workers", type=int,
                   help="Maximum parallel clips in each stage (default: automatic for preprocessing, 2 for detection)")
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
                   help="Store preprocessed clips here (default: %(default)s)")
    p.add_argument("--preprocess-cache", action="store_true",
                   help="Reuse saved metadata and default speedbreaker motion measurements. By default, refresh them.")
    p.add_argument("--motion-cache", type=Path, help="Reuse full-video motion measurements in this folder")
    p.add_argument("--hwaccel", choices=["d3d11va"], help="Use Windows GPU decoding for 8-bit motion analysis")
    p.add_argument("--sign-lookback", type=float, help="Search for signs this many seconds before each detected crossing")
    p.add_argument("--review", type=Path, help="Write an offline HTML video review page")
    return p


def build_args(parser: argparse.ArgumentParser, argv: Sequence[str] | None) -> argparse.Namespace:
    args = parser.parse_args(argv)
    requested = list(dict.fromkeys(n.strip() for n in args.detect.split(",")))
    unknown = set(requested) - AnalysisRun.detectors.keys()
    if unknown:
        parser.error(f"Unknown detector(s): {', '.join(sorted(unknown))}. Available: {', '.join(AnalysisRun.detectors)}")
    args.detect = [DetectorName(name) for name in requested]
    if args.sign_lookback is not None:
        required = {DetectorName.SPEEDBREAKER, DetectorName.WARNING_SIGN}
        if not (args.sign_lookback > 0 and required.issubset(args.detect)):
            parser.error("--sign-lookback requires a positive duration and both speedbreaker and warning_sign")
    if args.output.suffix.lower() != ".csv":
        parser.error("--output needs a .csv filename")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    return args


def check_prerequisites(inputs: Sequence[Path], parser: argparse.ArgumentParser) -> list[Path]:
    try:
        videos = FrontClips.resolve(inputs)
    except ValueError as err:
        parser.error(str(err))
    missing = [tool for tool in ("ffmpeg", "ffprobe", "exiftool") if shutil.which(tool) is None]
    if missing:
        parser.error(f"Missing from PATH: {', '.join(missing)}. See the README prerequisites for installation: "
                     "https://github.com/r-sniper/dashcam-event-detection#prerequisites")
    return videos


def preprocess_videos(videos: Sequence[Path], args: argparse.Namespace) -> list[ProcessedVideo]:
    preprocessor = VideoPreprocessor(args.work_dir, args.preprocess_cache)
    default_workers = min(8, cv2.getNumberOfCPUs())
    workers = min(len(videos), args.workers or default_workers)
    logger.info("Preprocessing (workers=%d)", workers)
    return thread_map(preprocessor.process, videos, max_workers=workers,
                      desc="Preprocessing", unit="video", file=sys.stdout)


def main(argv: Sequence[str] | None = None):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")
    logger.info("Parsing arguments")
    parser = build_parser()
    args = build_args(parser, argv)
    logger.info("Arguments parsed successfully")
    logger.info("Checking prerequisites")
    videos = check_prerequisites(args.inputs, parser)
    logger.info("Found %d %s", len(videos), "video" if len(videos) == 1 else "videos")

    cv2.setNumThreads(1)
    processed = preprocess_videos(videos, args)
    motion_folder = args.motion_cache or args.work_dir / "motion" / (args.hwaccel or "cpu")
    motion_cache = MotionCache(motion_folder, reuse=args.motion_cache is not None or args.preprocess_cache)
    run = AnalysisRun(args.detect, sign_lookback=args.sign_lookback, motion_cache=motion_cache, hwaccel=args.hwaccel)
    workers = min(len(processed), args.workers or 2)
    logger.info("Running detectors (workers=%d)", workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = run.scan_videos(processed, pool)

    events, clips = run.collect_results(results)
    logger.info("Saving results to %s", args.output)
    run.save(events, clips, args.output)
    SpeedbreakerCompilation.compile_output(args.output)
    if args.review:
        from visualize.review import ReviewPage
        logger.info("Writing review page to %s", args.review)
        ReviewPage.write(args.review, events, clips)
    logger.info("Scan complete. Saved %d detections to %s", len(events), args.output)
