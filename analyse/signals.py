"""Time-aligned camera motion and GPS measurements."""

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


@dataclass
class GpsTrace:
    seconds: NDArray[np.float64]
    speed: NDArray[np.float64]

    def at(self, seconds: float | NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.seconds.size:
            return np.full(np.shape(seconds), np.nan)
        speed = np.interp(seconds, self.seconds, self.speed)
        # Do not interpolate across a GPS outage.
        right = np.clip(np.searchsorted(self.seconds, seconds), 0, len(self.seconds) - 1)
        left = np.maximum(right - 1, 0)
        nearest = np.minimum(abs(seconds - self.seconds[right]), abs(seconds - self.seconds[left]))
        return np.where(nearest <= 2, speed, np.nan)


class CameraMotion:
    """Track scene features away from the bonnet and estimate background vertical movement."""

    width = 320
    height = 180
    fps = 10

    @classmethod
    def measure(cls, previous: NDArray[np.uint8], current: NDArray[np.uint8]) -> list[float]:
        points = cv2.goodFeaturesToTrack(previous, maxCorners=140, qualityLevel=0.01,
                                         minDistance=7, blockSize=5)
        if points is None or len(points) < 8:
            return [0., 0., 0.]
        tracked, status, errors = cv2.calcOpticalFlowPyrLK(
            previous, current, points, np.empty_like(points), winSize=(17, 17), maxLevel=2)
        keep = (status.ravel() == 1) & (errors.ravel() < 30)
        a, b = points[keep, 0], tracked[keep, 0]
        if len(a) < 8:
            return [0., 0., 0.]
        matrix, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                     ransacReprojThreshold=1.8, maxIters=150)
        if matrix is None:
            return [0., 0., 0.]
        # Coordinates are relative to the crop. Evaluate near the road vanishing point.
        x, y = cls.width / 2, cls.height * .50 - int(cls.height * .12)
        dy = matrix[1] @ [x, y, 1] - y
        return [dy, matrix[1, 0] * cls.width / 2, inliers.mean()]

    @classmethod
    def read(cls, video: Path, start: float = 0., end: float | None = None,
             hwaccel: str | None = None) -> NDArray[np.float64]:
        command = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "3"]
        filters = [f"fps={cls.fps}"]
        if hwaccel == "d3d11va":
            command += ["-hwaccel", "d3d11va", "-hwaccel_output_format", "d3d11"]
            filters += ["hwdownload", "format=nv12"]
        command += ["-ss", str(start), "-i", str(video)]
        if end is not None:
            command += ["-t", str(end - start)]
        filters.append(f"scale={cls.width}:{cls.height}:flags=fast_bilinear")
        command += ["-an", "-sn", "-vf", ",".join(filters),
                    "-filter_threads", "1", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"]
        motion = []
        previous = None
        with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
            frames = cast(BinaryIO, process.stdout)
            while data := frames.read(cls.height * cls.width):
                frame = np.frombuffer(data, np.uint8).reshape(cls.height, cls.width)
                current = frame[int(cls.height * .12):int(cls.height * .71)]
                if previous is not None:
                    motion.append(cls.measure(previous, current))
                previous = current
            if process.wait():
                raise RuntimeError(f"FFmpeg could not finish the motion scan of {video}")
        times = start + (np.arange(len(motion)) + .5) / cls.fps
        return np.column_stack([times, np.array(motion).reshape(-1, 3)])


class ScanWindows:
    """In fast mode inspect GPS troughs, clip boundaries and gaps in GPS coverage."""

    @staticmethod
    def guided(gps: GpsTrace, duration: float) -> list[tuple[float, float]]:
        seconds = np.arange(int(np.ceil(duration)), dtype=np.float64)
        speeds = gps.at(seconds)
        known = np.isfinite(speeds)
        if not known.any():
            return [(0., duration)]
        # Median filtering removes isolated zero-speed GPS fixes.
        smooth = median_filter(np.interp(seconds, seconds[known], speeds[known]), 3, mode="nearest")
        troughs, _ = find_peaks(-smooth, prominence=4, distance=4)
        centers = [int(i) for i in troughs if smooth[i] <= 36]
        for i in [0, len(seconds) - 1]:
            if smooth[i] <= 36:
                centers.append(i)
        keep = ~known
        for i in centers:
            keep[max(0, i - 9):i + 5] = True
        # Add context at the margins of each outage.
        for i in np.flatnonzero(np.diff(known.astype(int))):
            keep[max(0, i - 2):i + 4] = True
        edges = np.diff(np.r_[False, keep, False].astype(int))
        return [(float(a), min(float(b), duration))
                for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


class MotionCache:
    """Reuse expensive full-clip optical flow when scoring or annotations change."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def path(self, video: Path) -> Path:
        stat = video.stat()
        identity = f"flow-v1|{video.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        key = hashlib.sha256(identity.encode()).hexdigest()[:16]
        return self.folder / f"{video.stem}.{key}.npz"

    def store(self, video: Path, motion: NDArray[np.float64]):
        target = self.path(video)
        with tempfile.NamedTemporaryFile(dir=self.folder, suffix=".tmp", delete=False) as f:
            np.savez_compressed(f, motion=motion)
        os.replace(f.name, target)

    def read(self, video: Path, hwaccel: str | None = None) -> NDArray[np.float64]:
        target = self.path(video)
        if target.exists():
            with np.load(target) as data:
                return data["motion"]
        motion = CameraMotion.read(video, hwaccel=hwaccel)
        self.store(video, motion)
        return motion
