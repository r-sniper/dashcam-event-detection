"""Generate a standalone video index linking predictions to their source MP4s."""

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


class ReviewPage:
    @staticmethod
    def read(path: str | Path) -> list[dict[str, str]]:
        with Path(path).open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    @classmethod
    def write(cls, path: Path, events: Iterable[Mapping[str, Any]], clips: Sequence[Mapping[str, Any]],
              annotations: Sequence[Mapping[str, str]] = ()):
        sources = {c["filename"]: c["file_path"] for c in clips}
        entries = [dict(filename=r["filename"], url=Path(r["file_path"]).as_uri(), event=r["event"],
                        t=float(r["peak_s"]), score=r["score"], speed=r.get("speed_kmh"),
                        box=r.get("box", ""), range="", result="") for r in events]
        annotation_status = {}
        if annotations:
            from analyse.evaluate import GoldenSet
            predictions = defaultdict(list)
            for row in entries:
                if row["event"] == "speedbreaker":
                    predictions[row["filename"]].append(row["t"])
            details = GoldenSet(annotations).evaluate(predictions)
            result_by_peak = {(name, peak): status for name, _, _, peak, status in details if peak != ""}
            annotation_status = {serial: status for _, serial, _, _, status in details}
            for row in entries:
                if row["event"] == "speedbreaker":
                    row["result"] = result_by_peak.get((row["filename"], row["t"]), "unscored")
        for r in annotations:
            interval = r["speedbreaker time range"]
            entries.append(dict(filename=r["Filename"], url=Path(sources[r["Filename"]]).as_uri(),
                                event="manual_window", t=float(interval.split("-")[0]),
                                score="", speed="", box="", range=interval,
                                result=annotation_status.get(r["Sr no"], "")))
        entries.sort(key=lambda row: (row["filename"], row["t"]))
        template = Path(__file__).with_name("viewer.html.in").read_text(encoding="utf-8")
        data = json.dumps({"entries": entries, "clips": len(set(c["file_path"] for c in clips))},
                          ensure_ascii=True).replace("<", "\\u003c")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.replace("/*DATA*/null", data), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("detections", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--annotations", type=Path)
    args = p.parse_args()
    ReviewPage.write(args.output, ReviewPage.read(args.detections),
                     ReviewPage.read(args.detections.with_suffix(".clips.csv")),
                     ReviewPage.read(args.annotations) if args.annotations else ())


if __name__ == "__main__":
    main()
