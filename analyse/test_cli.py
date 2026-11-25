import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from .common import FrontClips
from .run import main


class ScanCliTests(unittest.TestCase):
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
