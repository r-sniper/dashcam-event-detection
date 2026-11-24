"""Optional visual candidates for upright Indian triangular warning signs."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .common import Detection, DetectorName
from .preprocess import ProcessedVideo


@dataclass
class WarningSignDetection(Detection):
    box: str  # Normalized x, y, width, height in the source frame.


@dataclass
class SignTrack:
    last: float
    count: int
    x: float
    y: float
    best: WarningSignDetection


class WarningSignDetector:
    name = DetectorName.WARNING_SIGN
    result_type = WarningSignDetection
    width, height, fps = 960, 540, 2

    @staticmethod
    def classify_face(face: NDArray[np.uint8]) -> tuple[str, float]:
        # Extract the central glyph before comparing its silhouette. Its printed
        # size and baseline position vary considerably between roadside signs.
        yy, xx = np.mgrid[:64, :64]
        inside = (yy > 18) & (yy < 54) & (abs(xx - 32) < (yy - 5) * .50 - 2)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        white = np.percentile(gray[inside], 80)
        dark = ((gray < white * .65) & inside).astype(np.uint8)
        n, components, stats, _ = cv2.connectedComponentsWithStats(dark)
        if n == 1:
            return "warning_sign_candidate", 0.
        k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x, y, w, h, area = stats[k]
        if not (w >= 10 and h >= 3 and 1.25 <= w/h <= 7 and area >= .68*dark.sum()):
            return "warning_sign_candidate", 0.
        glyph = components[y:y+h, x:x+w] == k
        # Use column thicknesses so a slightly tilted baseline is harmless.
        profile = np.interp(np.linspace(0, w-1, 41), np.arange(w), glyph.sum(axis=0))
        profile /= max(profile.max(), 1)
        axis = np.linspace(-1, 1, len(profile))
        best = {}
        for symbol, centers in [("possible_speedbreaker_sign", [0.]),
                                ("possible_rough_road_sign", [-.47, .47])]:
            error = 1.
            for width in (.38, .53, .70):
                arches = np.stack([np.sqrt(np.maximum(0, 1 - ((axis-c)/width)**2)) for c in centers])
                arch = arches.max(axis=0)
                for base in (.12, .30, .45):
                    error = min(error, float(np.abs(profile - (base + (1-base)*arch)).mean()))
            best[symbol] = 1 - error
        symbol = max(best, key=best.__getitem__)
        # A flat rectangle or an unresolved pictogram is only a generic warning.
        middle = profile[18:23].mean()
        sides = min(profile[7:15].max(), profile[26:34].max())
        resolved = (sides-middle > .12 if symbol == "possible_rough_road_sign"
                    else middle - max(profile[:7].mean(), profile[-7:].mean()) > .22)
        if best[symbol] < .78 or not resolved:
            return "warning_sign_candidate", 0.
        return symbol, best[symbol]

    def candidates(self, frame: NDArray[np.uint8], t: float) -> Iterator[tuple[float, float, WarningSignDetection]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        red = ((h < 12) | (h > 168)) & (s > 70) & (v > 55)
        red[:int(.06 * self.height)] = False
        red[int(.72 * self.height):] = False
        mask = cv2.morphologyEx(red.astype("uint8"), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (14 <= w <= 220 and .68 <= w / h <= 1.7):
                continue
            hull = cv2.convexHull(contour)
            polygon = cv2.approxPolyDP(hull, .065 * cv2.arcLength(hull, True), True)
            if len(polygon) != 3:
                continue
            top, *base = sorted(polygon[:, 0], key=lambda pt: pt[1])
            left, right = sorted(base, key=lambda pt: pt[0])
            if not (left[0] + .18*w < top[0] < right[0] - .18*w and abs(right[1]-left[1]) < .28*h):
                continue
            matrix = cv2.getAffineTransform(np.asarray([top, left, right], dtype=np.float32),
                                            np.asarray([[32, 1], [1, 62], [62, 62]], dtype=np.float32))
            face = cast(NDArray[np.uint8], cv2.warpAffine(frame, matrix, (64, 64)))
            inner = cv2.cvtColor(face[30:49, 24:40], cv2.COLOR_BGR2HSV)
            if ((inner[:, :, 1] < 85) & (inner[:, :, 2] > 100)).mean() < .50:
                continue
            label, similarity = self.classify_face(face)
            box = ",".join(f"{n:.4f}" for n in (x/self.width, y/self.height, w/self.width, h/self.height))
            yield (x + w/2)/self.width, (y + h/2)/self.height, self.result_type(label, t, round(similarity, 3), box)

    @staticmethod
    def approaches(crossings: Iterable[Detection], duration: float, lookback: float) -> list[tuple[float, float]]:
        windows = []
        for event in sorted(crossings, key=lambda e: e.peak_s):
            start = max(0., np.floor((event.peak_s-lookback)*2)/2)
            end = min(duration, np.ceil((event.peak_s+1)*2)/2)
            if windows and start <= windows[-1][1]:
                windows[-1] = (windows[-1][0], max(windows[-1][1], end))
            else:
                windows.append((start, end))
        return windows

    def scan_window(self, video: ProcessedVideo, start: float, end: float) -> list[WarningSignDetection]:
        tracks = []
        for frame in video.frames(start, end, self.width, self.height, self.fps):
            for x, y, event in self.candidates(frame.bgr, frame.timestamp_s):
                nearby = [tr for tr in tracks if 0 < frame.timestamp_s - tr.last <= 1.
                          and np.hypot(x - tr.x, y - tr.y) < .09]
                if nearby:
                    tr = min(nearby, key=lambda k: np.hypot(x-k.x, y-k.y))
                    tr.last, tr.x, tr.y = frame.timestamp_s, x, y
                    tr.count += 1
                    if event.score > tr.best.score:
                        tr.best = event
                else:
                    tracks.append(SignTrack(frame.timestamp_s, 1, x, y, event))
        return [tr.best for tr in tracks if tr.count >= 2]

    def detect(self, video: ProcessedVideo,
               windows: list[tuple[float, float]] | None = None) -> tuple[list[WarningSignDetection], float]:
        if not video.has_video_stream:
            raise ValueError(f"{video.path.name}: warning_sign requires a video stream")
        if windows is None:
            windows = [(0., video.duration)]
        events = [e for start, end in windows for e in self.scan_window(video, start, end)]
        return events, sum(end-start for start, end in windows)
