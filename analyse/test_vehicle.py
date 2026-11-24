import csv
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

from .common import DetectorName
from .preprocess import ProcessedVideo, VideoFrame
from .run import AnalysisRun, main
from .vehicle import IndianRegistration, VehicleDetection, VehicleDetector, VehicleTrack
from .warning_sign import WarningSignDetection, WarningSignDetector


class VehicleTests(unittest.TestCase):
    def test_clip_completion_logs_can_arrive_out_of_input_order(self):
        videos = [ProcessedVideo(Path(name), 1., True, [], None) for name in ("firstF.MP4", "secondF.MP4")]
        run = AnalysisRun([DetectorName.WARNING_SIGN])
        with (ThreadPoolExecutor(max_workers=2) as pool,
              patch.object(WarningSignDetector, "detect", return_value=([], 1.)),
              patch("analyse.run.as_completed", side_effect=reversed),
              self.assertLogs("analyse.run", level="INFO") as log):
            results = run.scan_videos(videos, pool)
        messages = [record.getMessage() for record in log.records]
        self.assertEqual([msg for msg in messages if msg.startswith("[")],
                         ["[1/2] Scanned secondF.MP4", "[2/2] Scanned firstF.MP4"])
        self.assertEqual([clip["filename"] for per_video in results for _, clip in per_video],
                         ["firstF.MP4", "secondF.MP4"])

    def test_default_output_for_single_and_combined_detectors(self):
        vehicle = DetectorName.VEHICLE
        crossing, sign = DetectorName.SPEEDBREAKER, DetectorName.WARNING_SIGN
        for names, path in [([vehicle], "analysis/vehicle/detections.csv"),
                            ([sign], "analysis/speedbreaker/detections.csv"),
                            ([crossing, sign], "analysis/speedbreaker/detections.csv"),
                            ([vehicle, crossing], "analysis/events.csv")]:
            with self.subTest(names=names):
                self.assertEqual(AnalysisRun.default_output(names), Path(path))

    def test_registration_formats(self):
        for number, kind in (("TS09AB1234", "state"), ("TG09A1234", "state"), ("DL9CAB1234", "state"),
                             ("24BH1234AB", "BH"), ("23BH0007A", "BH"), ("MHVAAB0001", "vintage"), ("TS09AB12", ""),
                             ("ZZ12AB1234", ""), ("24BH123OAB", "")):
            with self.subTest(number=number):
                self.assertEqual(IndianRegistration.format(number), kind)

    def test_consensus_keeps_the_timestamp_and_boxes_of_the_winning_read(self):
        box = Mock()
        first = VehicleDetection("vehicle", 1., .91, "car", 1., 1., "vehicle1", "plate1",
                                 "TS09AB1234", "state", .8)
        track = VehicleTrack(box, first, 1., 1.)
        track.observe(box, first)
        self.assertEqual(track.result().plate_number, "")
        self.assertEqual(track.result().plate_box, "plate1")
        for t, number, confidence in [(1.25, "TS09AB1284", .97), (1.5, "TS09AB1234", .87)]:
            track.observe(box, VehicleDetection("vehicle", t, .88, "car", t, t, str(t), str(t), number,
                                                "state", confidence))
        track.observe(box, VehicleDetection("vehicle", 2., .99, "car", 2., 2., "vehicle4"))
        best = track.result()
        self.assertEqual((best.start_s, best.peak_s, best.end_s), (1., 1.5, 2.))
        self.assertEqual((best.plate_number, best.plate_reads, best.plate_score), ("TS09AB1234", 2, .87))
        self.assertEqual((best.box, best.plate_box), ("1.5", "1.5"))

    @unittest.skipUnless(find_spec("open_image_models"), "vehicle extra required")
    def test_mixed_scan_associates_an_overlapping_plate_with_only_one_vehicle(self):
        from open_image_models.detection.core.base import DetectionResult

        detected = DetectionResult.from_detection_data
        car = detected((0, 0, 200, 120), .93, "car")
        truck = detected((1, 1, 199, 119), .79, "truck")
        bike = detected((100, 40, 180, 110), .87, "motorcycle")
        car_plate = detected((120, 85, 160, 100), .94, "License Plate")
        bike_plate = detected((20, 45, 60, 60), .97, "License Plate")
        frame = np.full((120, 200, 3), [5, 10, 20], dtype=np.uint8)
        vehicles, plates, ocr = Mock(), Mock(), Mock()
        vehicles.predict.side_effect = [[car, bike, truck], [bike, truck], [], [car]]
        plates.predict.side_effect = [[car_plate], [bike_plate], [bike_plate], [car_plate], []]
        ocr.return_value = Mock(txts=("24 BH", "1234.AB"), scores=(.93, .81))

        class SampleDetector(VehicleDetector):
            def __init__(self):
                self.vehicles, self.plates, self.ocr = vehicles, plates, ocr

        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "sampleF.MP4"
            path.touch()
            video = ProcessedVideo(path, 3., True, [], None)
            frames = [VideoFrame(t, frame) for t in (0., .25, 1., 2.)]
            sign = WarningSignDetection("warning_sign_candidate", 1., .82, "0.1,0.2,0.1,0.1")
            with (patch("analyse.run.VideoPreprocessor.process", return_value=video),
                  patch.dict("analyse.run.AnalysisRun.detectors", {VehicleDetector.name: SampleDetector}),
                  patch.object(WarningSignDetector, "detect", return_value=([sign], 3.)),
                  patch.object(ProcessedVideo, "native_frames", return_value=iter(frames)),
                  self.assertLogs("analyse.run", level="INFO") as log):
                main([str(path), "--detect", "vehicle,warning_sign", "--output", str(root / "events.csv"),
                      "--review", str(root / "events.html")])
            self.assertIn("Running detectors: vehicle, warning_sign", [r.getMessage() for r in log.records])
            with (root / "events.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            bike_row, = (row for row in rows if row["vehicle_type"] == "motorcycle")
            self.assertEqual((bike_row["plate_number"], bike_row["plate_reads"]), ("24BH1234AB", "2"))
            self.assertEqual(bike_row["plate_box"], "0.60000,0.70833,0.20000,0.12500")
            self.assertTrue(all(row["plate_box"] == "" for row in rows if row["vehicle_type"] == "car"))
            html = (root / "events.html").read_text(encoding="utf-8")
            review, _ = json.JSONDecoder().raw_decode(html.split("const data=", 1)[1])
            bike_entry, = (entry for entry in review["entries"] if entry["vehicle"] == "motorcycle")
            self.assertEqual((bike_entry["plate"], bike_entry["plateBox"], bike_entry["range"]),
                             (bike_row["plate_number"], bike_row["plate_box"], "0.0–0.25"))
            with (root / "events.clips.csv").open(newline="") as f:
                counts = {row["detector"]: row["events"] for row in csv.DictReader(f)}
            self.assertEqual(counts, {"vehicle": "3", "warning_sign": "1"})
        self.assertEqual(ocr.call_count, 2)
        np.testing.assert_array_equal(ocr.call_args.args[0][0, 0], [5, 10, 20])
