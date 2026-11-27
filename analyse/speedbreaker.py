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
    # Lowest threshold tied for best event-level F2 on the guide clips.
    threshold = 1.95
    separation_s = 5
    min_motion_samples = 3
    bounce_samples = 13
    vertical_clip_px = 20
    pitch_smoothing_sigma_samples = 0.8
    pitch_baseline_samples = 21
    quality_median_samples = 5
    full_weight_inlier_fraction = 0.50

    approach_offsets_s = (-8, -6, -4, -2)
    trough_offsets_s = range(7)
    trough_median_samples = 3
    crossing_offset_s = 2
    slow_speed_cutoff_kmh = 35
    slow_speed_range_kmh = 26
    full_brake_drop_kmh = 30
    full_crossing_brake_drop_kmh = 24

    slow_weight = 0.50
    brake_weight = 0.35
    crossing_brake_weight = 0.50
    missing_gps_prior = 0.40
    missing_crossing_brake_prior = 0.20

    def __init__(self, scan_mode: str = "full", *, motion_cache: MotionCache | None = None, hwaccel: str | None = None):
        self.scan_mode = scan_mode
        self.read_motion = motion_cache.read if motion_cache else CameraMotion.read
        self.hwaccel = hwaccel

    @classmethod
    def measure(cls, motion: NDArray[np.float64], gps: GpsTrace) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        t, vertical, _, quality = motion.T
        clipped = np.clip(vertical, -cls.vertical_clip_px, cls.vertical_clip_px)
        smoothed = gaussian_filter1d(clipped, cls.pitch_smoothing_sigma_samples)
        pitch = smoothed - median_filter(vertical, cls.pitch_baseline_samples, mode="nearest")
        up = np.maximum(0, maximum_filter1d(pitch, cls.bounce_samples))
        down = np.maximum(0, -minimum_filter1d(pitch, cls.bounce_samples))
        bounce = np.sqrt(up * down)
        # Weak RANSAC consensus often means the camera was following a moving object.
        inlier_fraction = median_filter(quality, cls.quality_median_samples)
        bounce *= np.clip(inlier_fraction / cls.full_weight_inlier_fraction, 0, 1)
        before = reduce(np.fmax, (gps.at(t + offset) for offset in cls.approach_offsets_s))
        # A single spurious zero between motorway-speed fixes is common in these exports.
        speeds = np.stack([gps.at(t + offset) for offset in cls.trough_offsets_s])
        width = cls.trough_median_samples
        trough = reduce(np.fmin, (np.median(speeds[i:i + width], axis=0) for i in range(len(speeds) - width + 1)))
        slow = np.clip((cls.slow_speed_cutoff_kmh - trough) / cls.slow_speed_range_kmh, 0, 1)
        brake = np.clip((before - trough) / cls.full_brake_drop_kmh, 0, 1)
        # A missing fix supplies no observed braking. Give motion-only events a neutral prior.
        crossing_drop = gps.at(t - cls.crossing_offset_s) - gps.at(t + cls.crossing_offset_s)
        crossing_brake = np.clip(crossing_drop / cls.full_crossing_brake_drop_kmh, 0, 1)
        prior = (cls.slow_weight * np.nan_to_num(slow, nan=cls.missing_gps_prior)
                 + cls.brake_weight * np.nan_to_num(brake, nan=cls.missing_gps_prior)
                 + cls.crossing_brake_weight * np.nan_to_num(crossing_brake, nan=cls.missing_crossing_brake_prior))
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
            if len(motion) < self.min_motion_samples:
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
        if len(motion) < self.min_motion_samples:
            raise ValueError(f"{video.path.name}: speedbreaker requires at least {self.min_motion_samples} camera-motion samples")
        if self.scan_mode == "full":
            return self.score_windows([motion], gps, video.duration), video.duration
        windows = ScanWindows.guided(gps, video.duration)
        chunks = (motion[(motion[:, 0] >= a) & (motion[:, 0] < b)] for a, b in windows)
        return self.score_windows(chunks, gps, video.duration), sum(b - a for a, b in windows)
