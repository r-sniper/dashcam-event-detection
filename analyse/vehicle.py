"""Track road vehicles and read Indian registration marks in their source-resolution crops."""

from __future__ import annotations

import logging
import math
import re
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .common import Detection, DetectorName
from .preprocess import ProcessedVideo, VideoFrame

if TYPE_CHECKING:
    from open_image_models.detection.core.base import BoundingBox, DetectionResult
    from rapidocr.utils.output import RapidOCROutput


logger = logging.getLogger(__name__)


@dataclass
class VehicleDetection(Detection):
    vehicle_type: str
    start_s: float
    end_s: float
    box: str
    plate_box: str = ""
    plate_number: str = ""
    plate_format: str = ""
    plate_score: float | None = None
    plate_reads: int = 0


class IndianRegistration:
    # Include old marks still on the road (e.g. AP/TS, OR, UA and DD).
    states = set("AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP "
                 "MZ NL OD OR PB PY RJ SK TN TR TS TG UA UK UP WB".split())

    @classmethod
    def format(cls, number: str) -> str:
        if re.fullmatch(r"[0-9]{2}BH[0-9]{4}[A-HJ-NP-Z]{1,2}", number):
            return "BH"
        if number[:2] in cls.states:
            if re.fullmatch(r"[A-Z]{2}VA[A-Z]{2}[0-9]{4}", number):
                return "vintage"
            if re.fullmatch(r"[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}", number):
                return "state"
        return ""


@dataclass
class VehicleTrack:
    last_box: BoundingBox
    best: VehicleDetection
    start: float
    end: float
    labels: Counter[str] = field(default_factory=Counter)
    reads: dict[str, tuple[int, float, VehicleDetection]] = field(default_factory=dict)

    def observe(self, box: BoundingBox, event: VehicleDetection):
        self.last_box = box
        self.end = event.peak_s
        self.labels[event.vehicle_type] += 1
        if (bool(event.plate_box), event.score) > (bool(self.best.plate_box), self.best.score):
            self.best = event
        if event.plate_score is not None:
            count, score, best = self.reads.get(event.plate_number, (0, event.plate_score, event))
            if event.plate_score > score:
                score, best = event.plate_score, event
            self.reads[event.plate_number] = count + 1, score, best

    def result(self) -> VehicleDetection:
        # Always retain the vehicle and any located plate, even when no text is reliable.
        best = replace(self.best, plate_number="", plate_format="", plate_score=None)
        if self.reads:
            votes, score, read = max(self.reads.values(), key=lambda item: item[:2])
            if votes >= 2 or score >= .95:
                best = replace(read, plate_reads=votes)
        return replace(best, start_s=self.start, end_s=self.end, vehicle_type=self.labels.most_common(1)[0][0])


class VehicleDetector:
    name = DetectorName.VEHICLE
    result_type = VehicleDetection
    fps = 4
    progress_interval_s = 5
    labels = {"car", "motorcycle", "bus", "truck"}

    def __init__(self):
        try:
            from onnxruntime import SessionOptions
            from open_image_models import create_detector
            from rapidocr import RapidOCR
        except ModuleNotFoundError as err:
            raise RuntimeError("Run vehicle scans with 'uv run --extra vehicle analyze.py ...'") from err

        options = SessionOptions()
        options.intra_op_num_threads = 3
        self.vehicles = create_detector("rf-detr-nano-384-coco", sess_options=options)
        self.plates = create_detector("yolo-v9-t-384-license-plate-end2end", conf_thresh=.45, sess_options=options)
        self.ocr = RapidOCR(params={
            "Global.use_cls": False, "Global.log_level": "error",
            "Det.limit_type": "max", "Det.limit_side_len": 512,
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        })

    @staticmethod
    def normalized(box: BoundingBox, frame: VideoFrame) -> str:
        height, width = frame.bgr.shape[:2]
        return ",".join(f"{n / size:.5f}" for n, size in zip(box.to_xywh(), (width, height, width, height)))

    def read_plate(self, image: NDArray[np.uint8], event: VehicleDetection):
        # Give the text detector enough pixels to separate stacked registration lines.
        scale = 96 / image.shape[0]
        enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        read = cast("RapidOCROutput", self.ocr(enlarged))
        if not read.txts or not read.scores:
            return
        number = re.sub(r"[\s.\-]", "", "".join(read.txts).upper()).removeprefix("IND")
        kind = IndianRegistration.format(number)
        confidence = min(read.scores)
        if kind and confidence >= .75:
            event.plate_number, event.plate_format = number, kind
            event.plate_score = round(confidence, 4)

    def observe(self, frame: VideoFrame, vehicle: DetectionResult,
                vehicles: list[DetectionResult]) -> VehicleDetection:
        box = vehicle.bounding_box
        t = frame.timestamp_s
        event = VehicleDetection(self.name.value, t, round(vehicle.confidence, 3), vehicle.label,
                                 t, t, self.normalized(box, frame))
        if box.width < 64 or box.height < 40:
            return event
        crop = frame.bgr[box.as_slices()]
        plates = cast("list[DetectionResult]", self.plates.predict(crop))
        for plate in sorted(plates, key=lambda p: p.confidence, reverse=True):
            local = plate.bounding_box.clamp(box.width, box.height)
            if local.is_empty or local.area > .45 * box.area or local.aspect_ratio < .85:
                continue
            plate_box = local.from_xywh(box.x1 + local.x1, box.y1 + local.y1, local.width, local.height)
            # A foreground bike or car can put its plate inside a larger vehicle's crop.
            if any(v.bounding_box.area < box.area and v.bounding_box.contains_point(*plate_box.center)
                   for v in vehicles):
                continue
            event.plate_box = self.normalized(plate_box, frame)
            if local.width >= 28 and local.height >= 10:
                self.read_plate(crop[local.as_slices()], event)
            break
        return event

    @staticmethod
    def match(active: list[VehicleTrack], vehicles: list[DetectionResult]) -> dict[int, VehicleTrack]:
        if not active or not vehicles:
            return {}
        # Cars, trucks and buses can change class between frames as the camera approaches.
        overlap = np.array([[track.last_box.iou(v.bounding_box)
                             if (track.best.vehicle_type == "motorcycle") == (v.label == "motorcycle") else 0.
                             for v in vehicles] for track in active])
        overlap[overlap < .20] = 0.
        rows, cols = linear_sum_assignment(overlap, maximize=True)
        return {int(j): active[i] for i, j in zip(rows, cols) if overlap[i, j] > 0.}

    def find_vehicles(self, frame: VideoFrame) -> list[DetectionResult]:
        detections = cast("list[DetectionResult]", self.vehicles.predict(frame.bgr))
        vehicles: list[DetectionResult] = []
        # RF-DETR returns class/box pairs in score order, sometimes two labels for the same box.
        for v in detections:
            if v.label in self.labels and not v.bounding_box.is_empty:
                if all(v.bounding_box.iou(other.bounding_box) < .85 for other in vehicles):
                    vehicles.append(v)
        return vehicles

    @classmethod
    def scan_frames(cls, video: ProcessedVideo) -> Iterator[VideoFrame]:
        frames = video.native_frames(cls.fps)
        if sys.stdout.isatty():
            with tqdm(frames, desc=f"{cls.name} {video.path.name}", total=math.ceil(video.duration * cls.fps),
                      unit="frame", file=sys.stdout, mininterval=1, miniters=1, leave=False) as progress:
                yield from progress
            return
        next_update = monotonic() + cls.progress_interval_s
        for count, frame in enumerate(frames, 1):
            yield frame
            now = monotonic()
            if now >= next_update:
                scanned = min(video.duration, frame.timestamp_s + 1 / cls.fps)
                logger.info("%s on %s | %.1f/%.1f s of footage (%.0f%%), %d frames processed",
                            cls.name, video.path.name, scanned, video.duration, 100 * scanned / video.duration, count)
                next_update = now + cls.progress_interval_s

    def detect(self, video: ProcessedVideo) -> tuple[list[VehicleDetection], float]:
        if not video.has_video_stream:
            raise ValueError(f"{video.path.name}: vehicle requires a video stream")
        active: list[VehicleTrack] = []
        completed = []
        for frame in self.scan_frames(video):
            t = frame.timestamp_s
            completed.extend(tr.result() for tr in active if t - tr.end > 1.)
            active = [tr for tr in active if t - tr.end <= 1.]
            vehicles = self.find_vehicles(frame)
            matched = self.match(active, vehicles)
            for i, vehicle in enumerate(vehicles):
                event = self.observe(frame, vehicle, vehicles)
                track = matched.get(i)
                if track is None:
                    track = VehicleTrack(vehicle.bounding_box, event, t, t)
                    active.append(track)
                track.observe(vehicle.bounding_box, event)
        completed.extend(tr.result() for tr in active)
        return completed, video.duration
