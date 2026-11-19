"""Time-aligned camera motion and GPS measurements."""

import csv
import hashlib
import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from telemetry_time import IST, parse_timestamp


@dataclass
class GpsTrace:
    seconds: np.ndarray
    speed: np.ndarray

    @classmethod
    def read(cls, video: Path):
        sidecar = video.parent / "telemetry" / f"{video.stem}.telemetry.csv"
        if sidecar.exists():
            content = sidecar.read_text(encoding="utf-8-sig")
        elif shutil.which("exiftool"):
            fmt = Path(__file__).resolve().parents[1] / "telemetry.fmt"
            command = ["exiftool", "-ee3", "-n", "-p", str(fmt), str(video)]
            try:
                content = subprocess.run(command, check=True, capture_output=True, text=True).stdout
            except (OSError, subprocess.CalledProcessError):
                content = ""
        else:
            content = ""
        try:
            start = datetime.strptime(video.stem[:16], "%Y_%m%d_%H%M%S").replace(tzinfo=IST)
        except ValueError:
            return cls(np.array([]), np.array([]))
        readings = {}
        for row in csv.DictReader(io.StringIO(content)):
            try:
                offset = (parse_timestamp(row["gps_timestamp_utc"]) - start).total_seconds()
                readings[offset] = float(row["speed_kmh"])
            except ValueError:
                continue
        times = np.array(sorted(readings))
        return cls(times, np.array([readings[t] for t in times]))

    def at(self, seconds):
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
    def measure(cls, previous, current):
        points = cv2.goodFeaturesToTrack(previous, maxCorners=140, qualityLevel=0.01,
                                         minDistance=7, blockSize=5)
        if points is None or len(points) < 8:
            return [0., 0., 0.]
        tracked, status, errors = cv2.calcOpticalFlowPyrLK(
            previous, current, points, None, winSize=(17, 17), maxLevel=2)
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
    def read(cls, video: Path, start=0., end=None, hwaccel=None):
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
            while data := process.stdout.read(cls.height * cls.width):
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
    def duration(video):
        args = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", str(video)]
        return float(subprocess.check_output(args, text=True))

    @staticmethod
    def guided(gps, duration):
        seconds = np.arange(int(np.ceil(duration)))
        speeds = gps.at(seconds)
        known = np.isfinite(speeds)
        if not known.any():
            return [(0., duration)]
        # Median filtering removes isolated zero-speed GPS fixes.
        smooth = median_filter(np.interp(seconds, seconds[known], speeds[known]), 3, mode="nearest")
        troughs, _ = find_peaks(-smooth, prominence=4, distance=4)
        centers = [i for i in troughs if smooth[i] <= 36]
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

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def path(self, video):
        stat = video.stat()
        identity = f"flow-v1|{video.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        key = hashlib.sha256(identity.encode()).hexdigest()[:16]
        return self.folder / f"{video.stem}.{key}.npz"

    def store(self, video, motion):
        target = self.path(video)
        with tempfile.NamedTemporaryFile(dir=self.folder, suffix=".tmp", delete=False) as f:
            np.savez_compressed(f, motion=motion)
        os.replace(f.name, target)

    def read(self, video, hwaccel=None):
        target = self.path(video)
        if target.exists():
            with np.load(target) as data:
                return data["motion"]
        motion = CameraMotion.read(video, hwaccel=hwaccel)
        self.store(video, motion)
        return motion
