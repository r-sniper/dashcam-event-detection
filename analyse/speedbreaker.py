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
from .signals import CameraMotion, GpsTrace, MotionCache, ScanWindows


PRE_ROLL_S = 2.0
POST_ROLL_S = 2.0


@dataclass
class SpeedbreakerDetection(Detection):
    bounce_px: float
    speed_kmh: float | None
    start_s: float
    end_s: float
    buffered_start_s: float
    buffered_end_s: float


class SpeedbreakerDetector:
    name = DetectorName.SPEEDBREAKER
    result_type = SpeedbreakerDetection
    # Lowest threshold tied for best event-level F2 on the 13 November guide clips.
    threshold = 1.95
    separation_s = 5
    bounce_samples = 13

    def __init__(self, scan_mode: str = "full", *, motion_cache: MotionCache | None = None, hwaccel: str | None = None):
        self.scan_mode = scan_mode
        self.read_motion = motion_cache.read if motion_cache else CameraMotion.read
        self.hwaccel = hwaccel

    @classmethod
    def measure(cls, motion: NDArray[np.float64], gps: GpsTrace) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        t, vertical, _, quality = motion.T
        smoothed = gaussian_filter1d(np.clip(vertical, -20, 20), .8)
        pitch = smoothed - median_filter(vertical, 21, mode="nearest")
        up = np.maximum(0, maximum_filter1d(pitch, cls.bounce_samples))
        down = np.maximum(0, -minimum_filter1d(pitch, cls.bounce_samples))
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

    @classmethod
    def event_range(cls, times: NDArray[np.float64], score: NDArray[np.float64],
                    peak: int, left: int, right: int) -> tuple[float, float]:
        first = last = peak
        while first > left and score[first - 1] >= cls.threshold:
            first -= 1
        while last < right and score[last + 1] >= cls.threshold:
            last += 1
        # Include the motion samples used to score the crossing. Times are sample centers.
        radius = cls.bounce_samples // 2
        half_sample = .5 / CameraMotion.fps
        start = times[max(left, first - radius)] - half_sample
        end = times[min(right, last + radius)] + half_sample
        return max(0., round(float(start), 2)), round(float(end), 2)

    def score_windows(self, chunks: Iterable[NDArray[np.float64]], gps: GpsTrace,
                      duration: float) -> list[SpeedbreakerDetection]:
        candidates = []
        for motion in chunks:
            if len(motion) < 3:
                continue
            bounce, score = self.measure(motion, gps)
            # Padding retains peaks at the beginning and end of a recording.
            peaks, _ = find_peaks(np.r_[-np.inf, score, -np.inf], height=self.threshold,
                                  distance=self.separation_s * CameraMotion.fps)
            peaks -= 1
            # Split sustained motion midway between distinct peaks.
            splits = (peaks[:-1] + peaks[1:] + 1) // 2
            for i, left, right in zip(peaks, np.r_[0, splits], np.r_[splits, len(motion)]):
                start, end = self.event_range(motion[:, 0], score, int(i), int(left), int(right) - 1)
                t = float(motion[i, 0])
                speed = gps.at(t)
                candidates.append(self.result_type(
                    event=self.name.value, peak_s=round(t, 2), score=round(float(score[i]), 3),
                    bounce_px=round(float(bounce[i]), 3),
                    speed_kmh=round(float(speed), 1) if np.isfinite(speed) else None,
                    start_s=start, end_s=end,
                    buffered_start_s=max(0., round(start - PRE_ROLL_S, 2)),
                    buffered_end_s=min(duration, round(end + POST_ROLL_S, 2)),
                ))
        # Adjacent search windows can contain the same bounce at their edges.
        kept = []
        for event in sorted(candidates, key=lambda event: -event.score):
            if all(abs(event.peak_s - other.peak_s) >= self.separation_s for other in kept):
                kept.append(event)
        return sorted(kept, key=lambda event: event.peak_s)

    def detect(self, video: ProcessedVideo) -> tuple[list[SpeedbreakerDetection], float]:
        gps = video.gps_trace
        if not np.any((gps.seconds >= 0) & (gps.seconds <= video.duration)):
            raise ValueError(f"{video.path.name}: speedbreaker requires GPS speed samples aligned to the clip")
        motion = self.read_motion(video.path, hwaccel=self.hwaccel, duration=video.duration)
        if len(motion) < 3:
            raise ValueError(f"{video.path.name}: speedbreaker requires at least three camera-motion samples")
        if self.scan_mode == "full":
            return self.score_windows([motion], gps, video.duration), video.duration
        windows = ScanWindows.guided(gps, video.duration)
        chunks = (motion[(motion[:, 0] >= a) & (motion[:, 0] < b)] for a, b in windows)
        return self.score_windows(chunks, gps, video.duration), sum(b - a for a, b in windows)
