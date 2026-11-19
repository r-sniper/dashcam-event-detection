"""One-to-one evaluation against manual clip-relative crossing intervals."""

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

MatchRow = tuple[str, str, str, float | str, str]


class GoldenSet:
    def __init__(self, rows: Iterable[Mapping[str, str]]):
        self.by_file: defaultdict[str, list[Mapping[str, str]]] = defaultdict(list)
        for row in rows:
            self.by_file[row["Filename"]].append(row)

    @classmethod
    def read(cls, path: str | Path) -> "GoldenSet":
        with Path(path).open(encoding="utf-8-sig", newline="") as f:
            return cls(csv.DictReader(f))

    def evaluate(self, predictions: Mapping[str, Sequence[float]]) -> list[MatchRow]:
        details: list[MatchRow] = []
        for name, labels in self.by_file.items():
            peaks = sorted(predictions[name])
            if any(r["speedbreaker time range"].endswith("-") for r in labels):
                for r in labels:
                    details.append((name, r["Sr no"], r["speedbreaker time range"], "", "unscored"))
                continue
            intervals = [tuple(map(float, r["speedbreaker time range"].split("-"))) for r in labels]
            valid = np.array([[a <= peak <= b for peak in peaks] for a, b in intervals], dtype=bool)
            matches = {}
            if peaks:
                ids, js = linear_sum_assignment(valid, maximize=True)
                matches = {i: j for i, j in zip(ids, js) if valid[i, j]}
            for i, r in enumerate(labels):
                details.append((name, r["Sr no"], r["speedbreaker time range"],
                                peaks[matches[i]] if i in matches else "", "hit" if i in matches else "miss"))
            for j, peak in enumerate(peaks):
                if j not in matches.values():
                    details.append((name, "", "", peak, "unmatched prediction"))
        return details

    @staticmethod
    def metrics(details: Sequence[MatchRow]) -> dict[str, float | int]:
        counts = {kind: sum(row[-1] == kind for row in details)
                  for kind in ("hit", "miss", "unmatched prediction", "unscored")}
        tp, fn, fp = counts["hit"], counts["miss"], counts["unmatched prediction"]
        return {**counts, "precision": tp / (tp + fp) if tp + fp else 0.,
                "recall": tp / (tp + fn) if tp + fn else 0.,
                "F1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    gold = GoldenSet.read(args.gold)
    with args.detections.with_suffix(".clips.csv").open(encoding="utf-8-sig", newline="") as f:
        coverage = {r["filename"] for r in csv.DictReader(f) if r["detector"] == "speedbreaker"}
    missing = gold.by_file.keys() - coverage
    if missing:
        parser.error(f"Missing scans for {', '.join(sorted(missing))}")
    predictions = defaultdict(list)
    with args.detections.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["event"] == "speedbreaker":
                predictions[row["filename"]].append(float(row["peak_s"]))
    detail = gold.evaluate(predictions)
    if args.output:
        with args.output.open("w", encoding="utf-8", newline="") as f:
            out = csv.writer(f)
            out.writerow(["filename", "annotation_id", "manual_range_s", "peak_s", "result"])
            out.writerows(detail)
    print(GoldenSet.metrics(detail))


if __name__ == "__main__":
    main()
