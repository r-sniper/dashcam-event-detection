"""Score individual crossings using pitch reversal and local GPS speed changes."""

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, median_filter, minimum_filter1d
from scipy.signal import find_peaks

from .common import Detection
from .signals import CameraMotion, GpsTrace, ScanWindows


class SpeedbreakerDetector:
    name = "speedbreaker"
    # Lowest threshold tied for best event-level F2 on the 13 November guide clips.
    threshold = 1.95
    separation_s = 5

    def __init__(self, scan_mode="full", cache=None, hwaccel=None):
        self.scan_mode = scan_mode
        self.cache = cache
        self.hwaccel = hwaccel

    @staticmethod
    def measure(motion, gps):
        t, vertical, _, quality = motion.T
        smoothed = gaussian_filter1d(np.clip(vertical, -20, 20), .8)
        pitch = smoothed - median_filter(vertical, 21, mode="nearest")
        up = np.maximum(0, maximum_filter1d(pitch, 13))
        down = np.maximum(0, -minimum_filter1d(pitch, 13))
        bounce = np.sqrt(up * down)
        # Weak RANSAC consensus often means the camera was following a moving object.
        bounce *= np.clip(median_filter(quality, 5) / .50, 0, 1)
        before = np.fmax.reduce([gps.at(t + offset) for offset in (-8, -6, -4, -2)])
        # A single spurious zero between motorway-speed fixes is common in these exports.
        speeds = [gps.at(t + offset) for offset in range(7)]
        trough = np.fmin.reduce([np.median(speeds[i:i + 3], axis=0) for i in range(5)])
        slow = np.clip((35 - trough) / 26, 0, 1)
        brake = np.clip((before - trough) / 30, 0, 1)
        # A missing fix supplies no observed braking. Give motion-only events a neutral prior.
        crossing_brake = np.clip((gps.at(t - 2) - gps.at(t + 2)) / 24, 0, 1)
        prior = (.50 * np.nan_to_num(slow, nan=.40) + .35 * np.nan_to_num(brake, nan=.40)
                 + .50 * np.nan_to_num(crossing_brake, nan=.20))
        return bounce, np.log1p(bounce) + prior

    def score_windows(self, chunks, gps):
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
                candidates.append(Detection(self.name, round(t, 2), round(float(score[i]), 3),
                                            round(float(bounce[i]), 3),
                                            round(float(speed), 1) if np.isfinite(speed) else None))
        # Adjacent search windows can contain the same bounce at their edges.
        kept = []
        for event in sorted(candidates, key=lambda event: -event.score):
            if all(abs(event.peak_s - other.peak_s) >= self.separation_s for other in kept):
                kept.append(event)
        return sorted(kept, key=lambda event: event.peak_s)

    def detect(self, video, duration):
        gps = GpsTrace.read(video)
        if self.scan_mode == "full" and self.cache is not None:
            return self.score_windows([self.cache.read(video, self.hwaccel)], gps), duration
        windows = (ScanWindows.guided(gps, duration) if self.scan_mode == "gps-first"
                   else [(0., duration)])
        chunks = (CameraMotion.read(video, a, b, self.hwaccel) for a, b in windows)
        return self.score_windows(chunks, gps), sum(b - a for a, b in windows)
