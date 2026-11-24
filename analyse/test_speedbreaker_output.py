import csv
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from .common import DetectorName
from .compilation import SourceStreams, SpeedbreakerCompilation
from .run import AnalysisRun
from .signals import CameraMotion, GpsTrace
from .speedbreaker import SpeedbreakerDetection, SpeedbreakerDetector
from .warning_sign import WarningSignDetection


class SpeedbreakerOutputTests(unittest.TestCase):
    def test_ranges_cover_scored_motion_and_split_sustained_crossings(self):
        motion = np.zeros((161, 4))
        motion[:, 0] = (np.arange(len(motion)) + .5) / CameraMotion.fps
        scores = np.zeros(len(motion))
        scores[:12] = scores[80:] = 2.
        scores[[0, 100, 160]] = 3.
        detector = SpeedbreakerDetector()
        gps = GpsTrace(np.array([0.]), np.array([15.]))
        with (patch.object(detector, "measure", return_value=(np.ones(len(motion)), scores)),
              patch("analyse.speedbreaker.PRE_ROLL_S", 3.),
              patch("analyse.speedbreaker.POST_ROLL_S", 1.25)):
            events = detector.score_windows([motion], gps, duration=16.15)
        self.assertEqual([(e.start_s, e.peak_s, e.end_s) for e in events],
                         [(0., .05, 1.8), (7.4, 10.05, 13.), (13., 16.05, 16.1)])
        self.assertEqual([(e.buffered_start_s, e.buffered_end_s) for e in events],
                         [(0., 3.05), (4.4, 14.25), (10., 16.15)])

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_saved_compilation_matches_csv_order_cuts_and_audio(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            results = []
            for name, color in [("bF", "blue"), ("aF", "red")]:
                source = root / f"{name}.mp4"
                frames = f"color=black:s=64x48:r=25:d=1,drawbox=color={color}:t=fill:enable='gte(t,0.5)'"
                audio = ["-f", "lavfi", "-i", "sine=sample_rate=16000:duration=1"] if name == "bF" else []
                subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", frames, *audio,
                                "-c:v", "mpeg4", str(source)], check=True)
                crossing = SpeedbreakerDetection("speedbreaker", .65, 2., 3., 15., start_s=.52, end_s=.8,
                                                 buffered_start_s=.36, buffered_end_s=.96)
                results.append([AnalysisRun.record(source, DetectorName.SPEEDBREAKER, "full", 1., 1., [crossing])])
            sign = WarningSignDetection("possible_speedbreaker_sign", .1, .8, "")
            sign_result = AnalysisRun.record(root / "aF.mp4", DetectorName.WARNING_SIGN, "full", 1., 1., [sign])
            results[-1].append(sign_result)
            output = root / "events.csv"
            events, clips = AnalysisRun.collect_results(results)
            AnalysisRun.save(events, clips, output)
            with output.open(newline="") as table:
                rows = list(csv.DictReader(table))
            columns = ("filename", "start_s", "end_s", "buffered_start_s", "buffered_end_s")
            self.assertEqual([tuple(row[key] for key in columns) for row in rows],
                             [("aF.mp4", "", "", "", ""), ("aF.mp4", "0.52", "0.8", "0.36", "0.96"),
                              ("bF.mp4", "0.52", "0.8", "0.36", "0.96")])

            video = output.with_suffix(".speedbreakers.mp4")
            self.assertFalse(video.exists())
            SpeedbreakerCompilation.compile_output(output)
            streams = SourceStreams.read(video)
            self.assertEqual([streams.video[k] for k in ("codec_name", "width", "height", "avg_frame_rate")],
                             ["mpeg4", 64, 48, "25/1"])
            assert streams.audio is not None
            self.assertEqual([streams.audio[k] for k in ("codec_name", "sample_rate", "channels")], ["aac", "16000", 1])
            decode = ["ffmpeg", "-v", "error", "-i", str(video)]
            rgb = subprocess.check_output([*decode, "-vf", "crop=2:2,scale=1:1", "-pix_fmt", "rgb24",
                                           "-f", "rawvideo", "pipe:1"])
            colors = np.frombuffer(rgb, dtype=np.uint8).reshape(-1, 3)
            self.assertEqual(len(colors), 30)
            self.assertTrue(np.all(colors[:4] < 16))
            self.assertTrue(np.all(colors[4:15, 0] > 240))
            self.assertTrue(np.all(colors[15:19] < 16))
            self.assertTrue(np.all(colors[19:, 2] > 240))
            pcm = subprocess.check_output([*decode, "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "pipe:1"])
            samples = np.frombuffer(pcm, dtype="<i2")
            self.assertEqual(np.abs(samples[800:2400]).max(), 0)
            self.assertGreater(np.abs(samples[6400:8000]).max(), 1000)

            muted_events, muted_clip = results[-1][0]
            AnalysisRun.save(muted_events, [muted_clip], output)
            SpeedbreakerCompilation.compile_output(output)
            self.assertIsNone(SourceStreams.read(video).audio)

            sign_events, sign_clip = sign_result
            AnalysisRun.save(sign_events, [sign_clip], output)
            self.assertTrue(video.exists())
            SpeedbreakerCompilation.compile_output(output)
            self.assertFalse(video.exists())
