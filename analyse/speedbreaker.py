"""Score individual crossings using pitch reversal and local GPS speed changes."""

from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, median_filter, minimum_filter1d
from scipy.signal import find_peaks

from .common import Detection, DetectorName
from .preprocess import ProcessedVideo
from .signals import CameraMotion, GpsTrace, ScanWindows


@dataclass
class SpeedbreakerDetection(Detection):
    bounce_px: float
    speed_kmh: float | None


class SpeedbreakerDetector:
    name = DetectorName.SPEEDBREAKER
    result_type = SpeedbreakerDetection
    # Lowest threshold tied for best event-level F2 on the 13 November guide clips.
    threshold = 1.95
    separation_s = 5

    def __init__(self, scan_mode: str = "full"):
        self.scan_mode = scan_mode

    @staticmethod
    def measure(motion: NDArray[np.float64], gps: GpsTrace) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        t, vertical, _, quality = motion.T
        smoothed = gaussian_filter1d(np.clip(vertical, -20, 20), .8)
        pitch = smoothed - median_filter(vertical, 21, mode="nearest")
        up = np.maximum(0, maximum_filter1d(pitch, 13))
        down = np.maximum(0, -minimum_filter1d(pitch, 13))
        bounce = np.sqrt(up * down)
        # Weak RANSAC consensus often means the camera was following a moving object.
        bounce *= np.clip(median_filter(quality, 5) / .50, 0, 1)
        before = reduce(np.fmax, (gps.at(t + offset) for offset in (-8, -6, -4, -2)))
        # A single spurious zero between motorway-speed fixes is common in these exports.
        speeds = np.stack([gps.at(t + offset) for offset in range(7)])
        trough = reduce(np.fmin, (np.median(speeds[i:i + 3], axis=0) for i in range(5)))
        slow = np.clip((35 - trough) / 26, 0, 1)
        brake = np.clip((before - trough) / 30, 0, 1)
        # A missing fix supplies no observed braking. Give motion-only events a neutral prior.
        crossing_brake = np.clip((gps.at(t - 2) - gps.at(t + 2)) / 24, 0, 1)
        prior = (.50 * np.nan_to_num(slow, nan=.40) + .35 * np.nan_to_num(brake, nan=.40)
                 + .50 * np.nan_to_num(crossing_brake, nan=.20))
        return bounce, np.log1p(bounce) + prior

    def score_windows(self, chunks: Iterable[NDArray[np.float64]], gps: GpsTrace) -> list[SpeedbreakerDetection]:
        candidates = []
        for motion in chunks:
            if len(motion) < 3:
                continue
            bounce, score = self.measure(motion, gps)
            # Padding retains peaks at the beginning and end of a recording.
            peaks, _ = find_peaks(np.r_[-np.inf, score, -np.inf], height=self.threshold,
                                  distance=self.separation_s * CameraMotion.fps)
            for i in peaks - 1:
                t = float(motion[i, 0])
                speed = gps.at(t)
                candidates.append(self.result_type(
                    event=self.name.value, peak_s=round(t, 2), score=round(float(score[i]), 3),
                    bounce_px=round(float(bounce[i]), 3),
                    speed_kmh=round(float(speed), 1) if np.isfinite(speed) else None,
                ))
        # Adjacent search windows can contain the same bounce at their edges.
        kept = []
        for event in sorted(candidates, key=lambda event: -event.score):
            if all(abs(event.peak_s - other.peak_s) >= self.separation_s for other in kept):
                kept.append(event)
        return sorted(kept, key=lambda event: event.peak_s)

    def detect(self, video: ProcessedVideo) -> tuple[list[SpeedbreakerDetection], float]:
        motion = video.motion
        if motion is None or len(motion) < 3:
            raise ValueError(f"{video.path.name}: speedbreaker requires at least three camera-motion samples")
        gps = video.gps_trace
        if not np.any((gps.seconds >= 0) & (gps.seconds <= video.duration)):
            raise ValueError(f"{video.path.name}: speedbreaker requires GPS speed samples aligned to the clip")
        if self.scan_mode == "full":
            return self.score_windows([motion], gps), video.duration
        windows = ScanWindows.guided(gps, video.duration)
        chunks = (motion[(motion[:, 0] >= a) & (motion[:, 0] < b)] for a, b in windows)
        return self.score_windows(chunks, gps), sum(b - a for a, b in windows)
