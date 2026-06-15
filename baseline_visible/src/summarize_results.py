from __future__ import annotations

import csv
from pathlib import Path


ONLY_VISIBLE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = ONLY_VISIBLE_DIR / "workspace"


def metric(row: dict[str, str], name: str) -> float:
    return float(row.get(name, "nan"))


def main() -> None:
    if not WORKSPACE_DIR.exists():
        print("No baseline_visible workspace exists yet.")
        return

    summaries: list[tuple[str, int, float, float, float, float]] = []
    for results_csv in sorted(WORKSPACE_DIR.glob("*/results.csv")):
        with results_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue

        best_row = max(rows, key=lambda row: metric(row, "metrics/mAP50(B)"))
        summaries.append(
            (
                results_csv.parent.name,
                len(rows),
                metric(best_row, "metrics/precision(B)"),
                metric(best_row, "metrics/recall(B)"),
                metric(best_row, "metrics/mAP50(B)"),
                metric(best_row, "metrics/mAP50-95(B)"),
            )
        )

    if not summaries:
        print("No completed baseline_visible training epochs found.")
        return

    print("run,epochs,precision,recall,mAP50,mAP50-95")
    for values in summaries:
        name, epochs, precision, recall, map50, map5095 = values
        print(
            f"{name},{epochs},{precision:.6f},{recall:.6f},"
            f"{map50:.6f},{map5095:.6f}"
        )


if __name__ == "__main__":
    main()

