import argparse
import csv
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import fields
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any, TypedDict

import cv2
from tqdm.contrib.concurrent import thread_map

from .common import Detection, DetectorName, EventRow, FrontClips
from .detector import Detector, DetectorOptions
from .preprocess import DEFAULT_WORK_DIR, ProcessedVideo, VideoPreprocessor
from .signals import MotionCache
from .speedbreaker import SpeedbreakerDetector
from .warning_sign import WarningSignDetector


logger = logging.getLogger(__name__)


class ClipRow(TypedDict):
    filename: str
    file_path: str
    detector: DetectorName
    scan_mode: str
    duration_s: float
    inspected_s: float
    events: int


ScanResult = tuple[Sequence[Detection], ClipRow]
DETECTORS: dict[str, type[Detector[Any]]] = {
    detector.name: detector for detector in (SpeedbreakerDetector, WarningSignDetector)
}


class AnalysisRun:
    clip_fields = ["filename", "file_path", "detector", "scan_mode", "duration_s", "inspected_s", "events"]

    def __init__(self, names: Sequence[DetectorName], options: DetectorOptions = DetectorOptions()):
        self.output = options.output
        configured = {factory.name: factory(options) for factory in DETECTORS.values()}
        result_fields = (field.name for detector in configured.values() for field in fields(detector.result_type))
        self.event_fields = list(dict.fromkeys(["encounter_id", "filename", "file_path", *result_fields]))
        order = TopologicalSorter({name: configured[name].dependencies for name in names})
        self.selected = [configured[name] for name in order.static_order()]
        self.inactive = [detector for name, detector in configured.items() if name not in names]

    def run_detectors(self, video: ProcessedVideo) -> list[ScanResult]:
        results = []
        previous: dict[DetectorName, Sequence[Detection]] = {}
        for detector in self.selected:
            events, inspected = detector.detect(video, previous)
            previous[detector.name] = events
            clip = ClipRow(filename=video.path.name, file_path=str(video.path.resolve()), detector=detector.name,
                           scan_mode=detector.scan_mode, duration_s=round(video.duration, 3),
                           inspected_s=round(inspected, 3), events=len(events))
            results.append((events, clip))
        return results

    def _scan_video(self, video: ProcessedVideo) -> list[ScanResult] | None:
        try:
            return self.run_detectors(video)
        except Exception as err:
            logger.warning("Skipping %s after detection failed (%s)", video.path, err)
            return None

    def scan_videos(self, videos: Sequence[ProcessedVideo], workers: int) -> list[list[ScanResult]]:
        results = thread_map(self._scan_video, videos, max_workers=workers,
                             desc="Detection", unit="video", file=sys.stdout)
        return [result for result in results if result is not None]

    @staticmethod
    def assign_ids(events: Sequence[EventRow], clips: Sequence[ClipRow],
                   owners: Iterable[type[Detector[Any]]] | None = None) -> list[EventRow]:
        if owners is None:
            owners = (DETECTORS.get(row["event"], Detector) for row in events)
        entries = sorted(zip(events, owners, strict=True),
                         key=lambda entry: (entry[0]["file_path"], entry[0]["peak_s"], entry[0]["event"]))
        durations = {clip["file_path"]: clip["duration_s"] for clip in clips}
        previous: dict[type[Detector[Any]], tuple[EventRow, int]] = {}
        next_id = 1
        for event, owner in entries:
            last = previous.get(owner)
            if last and owner.continues_encounter(last[0], event, durations[last[0]["file_path"]]):
                encounter_id = last[1]
            else:
                encounter_id = next_id
                next_id += 1
            event["encounter_id"] = encounter_id
            previous[owner] = event, encounter_id
        return [event for event, _ in entries]

    @staticmethod
    def write_table(path: Path, fields: Collection[str], rows: Iterable[Mapping[str, object]]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", newline="",
                                         suffix=".tmp", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(f.name, path)

    def write_detector_outputs(self, by_detector: Mapping[DetectorName, list[Detection]]):
        for detector in self.selected:
            events = by_detector[detector.name]
            for event in events:
                detector.event_output(event)
            detector.final_output(events)
        for detector in self.inactive:
            detector.clear_output()

    def save_results(self, results: Iterable[list[ScanResult]]) -> tuple[list[EventRow], list[ClipRow]]:
        clips = []
        by_detector: dict[DetectorName, list[Detection]] = {detector.name: [] for detector in self.selected}
        for per_video in results:
            for events, clip in per_video:
                clips.append(clip)
                by_detector[clip["detector"]].extend(events)
        rows = []
        owners = []
        for detector in self.selected:
            found = by_detector[detector.name]
            found.sort(key=lambda event: (str(event.file_path), event.peak_s, event.event))
            for event in found:
                owners.append(type(detector))
                rows.append(event.as_row())
        rows = self.assign_ids(rows, clips, owners)
        self.write_table(self.output, self.event_fields, rows)
        self.write_table(self.output.with_suffix(".clips.csv"), self.clip_fields, clips)

        self.write_detector_outputs(by_detector)
        return rows, clips


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Find events in a list or folder of Viofo front-camera MP4s.")
    p.add_argument("inputs", type=Path, nargs="+", help="Front MP4s, or folders containing them")
    available = ", ".join(DETECTORS)
    p.add_argument("--detect", default=DetectorName.SPEEDBREAKER.value, help=f"Comma-separated: {available}")
    # Accept the explicit full mode used by earlier CLI commands.
    p.add_argument("--scan-mode", choices=("full",), help=argparse.SUPPRESS)
    p.add_argument("--output", type=Path, default=DetectorOptions.output)
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
    unknown = set(requested) - DETECTORS.keys()
    if unknown:
        parser.error(f"Unknown detector(s): {', '.join(sorted(unknown))}. Available: {', '.join(DETECTORS)}")
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
    # Show cache hits even if they all finish within one redraw interval.
    return thread_map(preprocessor.process, videos, max_workers=workers, miniters=1, mininterval=0,
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
    options = DetectorOptions(output=args.output, sign_lookback=args.sign_lookback,
                              motion_cache=motion_cache, hwaccel=args.hwaccel)
    run = AnalysisRun(args.detect, options)
    workers = min(len(processed), args.workers or 2)
    logger.info("Running detectors (workers=%d)", workers)
    results = run.scan_videos(processed, workers)

    logger.info("Saving results to %s", args.output)
    events, clips = run.save_results(results)
    if args.review:
        from visualize.review import ReviewPage
        logger.info("Writing review page to %s", args.review)
        ReviewPage.write(args.review, events, clips)
    logger.info("Video scan complete (%d succeeded, %d failed). Saved %d detections to %s",
                len(results), len(processed) - len(results), len(events), args.output)
    logger.info("If something looks wrong, include the relevant log messages in a GitHub issue at "
                "https://github.com/r-sniper/dashcam-event-detection/issues/new")
