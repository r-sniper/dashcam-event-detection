import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from .preprocess import FFprobeResult, ProcessedVideo, VideoPreprocessor
from .run import main
from .speedbreaker import SpeedbreakerDetector
from .warning_sign import WarningSignDetector


class PreprocessingTests(unittest.TestCase):
    metadata = {"format": {"duration": "10"}, "streams": [{"codec_type": "video", "width": 2560}]}
    telemetry = ("gps_timestamp_utc,latitude_deg,longitude_deg,speed_kmh,heading_deg\n"
                 "2026:11:13 01:14:14Z,17.61,78.13,42.5,356.6\n")

    def test_run_preprocesses_overlapping_inputs_once_for_both_detectors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "exampleF.MP4"
            video.touch()
            processed = ProcessedVideo(video, 10., True, [], None)
            with (patch.object(VideoPreprocessor, "process", return_value=processed) as preprocess,
                  patch.object(SpeedbreakerDetector, "detect", return_value=([], 10.)) as crossings,
                  patch.object(WarningSignDetector, "detect", return_value=([], 10.)) as signs):
                main([str(root), str(video), "--detect", "speedbreaker,warning_sign", "--output", str(root / "events.csv")])
            preprocess.assert_called_once_with(video.resolve())
            crossings.assert_called_once_with(processed)
            signs.assert_called_once_with(processed)

    def test_refresh_reuse_and_source_change(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "2026_1113_064412_002414F.MP4"
            video.touch()
            fresh = VideoPreprocessor(root)
            reuse = VideoPreprocessor(root, reuse_cache=True)
            outputs = [json.dumps(self.metadata), self.telemetry] * 3
            with (patch("analyse.preprocess.subprocess.check_output", side_effect=outputs) as extract,
                  patch("analyse.preprocess.CameraMotion.read", return_value=np.ones((5, 4))) as flow):
                fresh.process(video)
                fresh.process(video)
                cached = reuse.process(video)
                self.assertEqual(extract.call_count, 4)
                self.assertEqual(flow.call_count, 2)
                self.assertEqual(cached.duration, 10.)
                self.assertIsInstance(cached.duration, float)
                self.assertTrue(cached.has_video_stream)
                self.assertEqual(cached.debug_metadata, self.metadata)
                self.assertEqual(cached.gps_trace.at(2), 42.5)
                self.assertEqual(cached.gps_samples[0].longitude_deg, 78.13)
                np.testing.assert_array_equal(cached.motion, np.ones((5, 4)))
                video.write_bytes(b"replacement")
                reuse.process(video)
                self.assertEqual(flow.call_count, 3)

    def test_probe_handles_missing_duration_and_audio_only_clips(self):
        video = Path("exampleF.MP4")
        for metadata in ({}, {"format": {"duration": "N/A"}}):
            with self.subTest(metadata=metadata), patch("analyse.preprocess.subprocess.check_output",
                                                       return_value=json.dumps(metadata)):
                with self.assertRaisesRegex(ValueError, "exampleF.MP4.*duration"):
                    FFprobeResult.read(video)
        audio_only = {"format": {"duration": "1.25"}, "streams": [{"codec_type": "audio"}]}
        with patch("analyse.preprocess.subprocess.check_output", return_value=json.dumps(audio_only)):
            probe = FFprobeResult.read(video)
        self.assertEqual(probe.duration, 1.25)
        self.assertFalse(probe.has_video_stream)

    def test_detectors_require_only_their_inputs(self):
        video = ProcessedVideo(path=Path("exampleF.MP4"), duration=10., has_video_stream=True,
                               gps_samples=[], motion=None)
        with self.assertRaisesRegex(ValueError, "exampleF.MP4.*camera-motion"):
            SpeedbreakerDetector().detect(video)
        video.motion = np.ones((5, 4))
        with self.assertRaisesRegex(ValueError, "exampleF.MP4.*GPS speed"):
            SpeedbreakerDetector().detect(video)
        with patch.object(ProcessedVideo, "frames", return_value=iter(())) as frames:
            self.assertEqual(WarningSignDetector().detect(video), ([], 10.))
            frames.assert_called_once()
        video.has_video_stream = False
        video.debug_metadata = self.metadata
        with self.assertRaisesRegex(ValueError, "warning_sign requires a video stream"):
            WarningSignDetector().detect(video)
