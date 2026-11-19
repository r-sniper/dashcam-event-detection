import argparse
import csv
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2

from .common import Detection, FrontClips
from .signals import MotionCache, ScanWindows
from .speedbreaker import SpeedbreakerDetector
from .warning_sign import WarningSignDetector


class AnalysisRun:
    detectors = {"speedbreaker": SpeedbreakerDetector, "warning_sign": WarningSignDetector}
    event_fields = ["encounter_id", "filename", "file_path", *Detection.__dataclass_fields__]
    clip_fields = ["filename", "file_path", "detector", "scan_mode", "duration_s", "inspected_s", "events"]

    def __init__(self, names, scan_mode, cache=None, hwaccel=None, sign_lookback=None):
        self.scan_mode = scan_mode
        self.sign_lookback = sign_lookback
        if sign_lookback is not None:
            names = sorted(names, key=lambda name: name != "speedbreaker")
        self.selected = [self.detectors[name](scan_mode, cache, hwaccel) if name == "speedbreaker" else self.detectors[name]()
                         for name in names]

    @staticmethod
    def record(video, detector, mode, duration, inspected, detections):
        source = {"filename": video.name, "file_path": str(video.resolve())}
        events = [dict(source, **asdict(item)) for item in detections]
        clip = dict(source, detector=detector, scan_mode=mode, duration_s=round(duration, 3),
                    inspected_s=round(inspected, 3), events=len(events))
        return events, clip

    def scan(self, video):
        duration = ScanWindows.duration(video)
        results = []
        crossings = []
        for detector in self.selected:
            mode = self.scan_mode if detector.name == "speedbreaker" else "full"
            if detector.name == "warning_sign" and self.sign_lookback is not None:
                windows = detector.approaches(crossings, duration, self.sign_lookback)
                events, inspected = detector.detect(video, duration, windows)
                mode = f"approach-{self.sign_lookback:g}s"
            else:
                events, inspected = detector.detect(video, duration)
            if detector.name == "speedbreaker":
                crossings = events
            results.append(self.record(video, detector.name, mode, duration, inspected, events))
        return results

    @staticmethod
    def assign_ids(events, clips):
        durations = {row["file_path"]: row["duration_s"] for row in clips}
        previous = None
        next_id = 0
        for event in sorted(events, key=lambda e: (e["file_path"], e["peak_s"], e["event"])):
            same = False
            if event["event"] == "speedbreaker" and previous is not None:
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
                        same = abs(seam_gap) <= 1 and 0 <= gap <= 2.5
            if same:
                event["encounter_id"] = previous["encounter_id"]
            else:
                next_id += 1
                event["encounter_id"] = next_id
            if event["event"] == "speedbreaker":
                previous = event

    @staticmethod
    def write_table(path, fields, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", newline="",
                                         suffix=".tmp", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(f.name, path)

    @classmethod
    def save(cls, results, output):
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
    p.add_argument("--detect", default="speedbreaker", help="Comma-separated: speedbreaker,warning_sign")
    p.add_argument("--scan-mode", choices=("full", "gps-first"), default="full",
                   help="Full scans all footage. GPS-first scans around speed troughs and GPS gaps.")
    p.add_argument("--output", type=Path, default=Path("analysis/speedbreaker/detections.csv"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--motion-cache", type=Path, help="Reuse full-video motion measurements in this folder")
    p.add_argument("--hwaccel", choices=["d3d11va"], help="Use Windows GPU decoding for 8-bit motion analysis")
    p.add_argument("--sign-lookback", type=float, help="Search for signs this many seconds before each detected crossing")
    p.add_argument("--review", type=Path, help="Write an offline HTML video review page")
    args = p.parse_args()
    names = list(dict.fromkeys(n.strip() for n in args.detect.split(",")))
    unknown = set(names) - AnalysisRun.detectors.keys()
    if unknown:
        p.error(f"Unknown detector(s): {', '.join(sorted(unknown))}. Available: {', '.join(AnalysisRun.detectors)}")
    if args.sign_lookback is not None and (args.sign_lookback <= 0 or not {"speedbreaker", "warning_sign"}.issubset(names)):
        p.error("--sign-lookback requires a positive duration and both speedbreaker and warning_sign")
    if args.output.suffix.lower() != ".csv" or args.workers < 1:
        p.error("--output needs a .csv filename and --workers must be positive")
    cv2.setNumThreads(1)
    run = AnalysisRun(names, args.scan_mode, MotionCache(args.motion_cache) if args.motion_cache else None,
                      args.hwaccel, args.sign_lookback)
    try:
        videos = FrontClips.resolve(args.inputs)
    except ValueError as err:
        p.error(str(err))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        def results():
            for i, per_video in enumerate(pool.map(run.scan, videos), 1):
                print(f"[{i}/{len(videos)}] {videos[i-1].name}", flush=True)
                yield per_video
        events, clips = run.save(results(), args.output)
    if args.review:
        from visualize.review import ReviewPage
        ReviewPage.write(args.review, events, clips)
    print(f"{len(events)} detections across {len(videos)} clips. Saved {args.output}")
