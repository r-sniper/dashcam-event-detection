import csv
import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from .common import FrontClips
from .preprocess import GpsSample, ProcessedVideo
from .run import main
from .signals import CameraMotion


class ScanCliTests(unittest.TestCase):
    def test_detection_continues_after_a_clip_with_missing_gps(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            missing = ProcessedVideo(root / "missingF.MP4", 10., True, [])
            good = ProcessedVideo(root / "goodF.MP4", 10., True, [GpsSample("", 0., None, None, 20., None)])
            missing.path.touch()
            good.path.touch()
            output = root / "events.csv"
            with (patch("analyse.run.shutil.which", return_value="installed"),
                  patch("analyse.run.preprocess_videos", return_value=[missing, good]),
                  patch.object(CameraMotion, "read", return_value=np.zeros((5, 4))),
                  self.assertLogs("analyse.run", level="INFO") as logs):
                main([str(missing.path), str(good.path), "--work-dir", directory, "--output", str(output)])

            with output.with_suffix(".clips.csv").open(newline="") as table:
                self.assertEqual([(r["filename"], r["events"]) for r in csv.DictReader(table)], [(good.path.name, "0")])
            warnings = [r.getMessage() for r in logs.records if r.levelname == "WARNING"]
            self.assertEqual(len(warnings), 1)
            self.assertIn(missing.path.name, warnings[0])
            self.assertIn("GPS speed", warnings[0])
            self.assertIn("1 succeeded, 1 failed", logs.output[-2])
            self.assertIn("https://github.com/r-sniper/dashcam-event-detection/issues/new", logs.output[-1])

    def test_invalid_arguments_fail_before_clip_discovery(self):
        cases = [
            (["--detect", "missing"], "Unknown detector(s): missing"),
            (["--sign-lookback", "10"], "both speedbreaker and warning_sign"),
            (["--detect", "speedbreaker,warning_sign", "--sign-lookback", "0"], "positive duration"),
            (["--output", "events.txt"], "--output needs a .csv filename"),
            (["--workers", "0"], "--workers must be positive"),
        ]
        with patch.object(FrontClips, "resolve") as resolve:
            for options, message in cases:
                with self.subTest(options=options), redirect_stderr(io.StringIO()) as error:
                    with self.assertRaises(SystemExit) as exit_status:
                        main(["clips", *options])
                    self.assertEqual(exit_status.exception.code, 2)
                    self.assertIn(message, error.getvalue())
            resolve.assert_not_called()

    def test_missing_tools_are_listed_with_installation_instructions(self):
        with TemporaryDirectory() as directory:
            video = Path(directory) / "exampleF.MP4"
            video.touch()
            installed = {"ffprobe": "installed"}
            with (patch("analyse.run.shutil.which", side_effect=installed.get),
                  redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit) as exit_status):
                main([str(video)])
            self.assertEqual(exit_status.exception.code, 2)
            self.assertIn("Missing from PATH: ffmpeg, exiftool.", error.getvalue())
            self.assertIn("dashcam-event-detection#prerequisites", error.getvalue())

    def test_empty_folder_reports_missing_clips(self):
        with (TemporaryDirectory() as directory, redirect_stderr(io.StringIO()) as error,
              self.assertRaises(SystemExit) as exit_status):
            main([directory])
        self.assertEqual(exit_status.exception.code, 2)
        self.assertIn("No front-camera *F.MP4 files found", error.getvalue())
