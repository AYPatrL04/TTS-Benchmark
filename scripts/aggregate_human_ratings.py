from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


RATING_FIELDS = (
    "intelligibility_1_5",
    "naturalness_1_5",
    "emotion_match_1_5",
    "overall_acceptability_1_5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed blind TTS ratings by clip.")
    parser.add_argument("--ratings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def parse_rating(row: dict[str, str], field: str) -> float:
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError):
        return math.nan
    if value < 1.0 or value > 5.0:
        raise ValueError(f"{row.get('blind_id', '')} {field} must be between 1 and 5")
    return value


def main() -> None:
    args = parse_args()
    with args.ratings.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("blind_id") and row.get("rater_id"):
            groups[row["blind_id"]].append(row)

    output_rows = []
    for blind_id, items in sorted(groups.items()):
        result = {"blind_id": blind_id, "rater_count": str(len({row['rater_id'] for row in items}))}
        for field in RATING_FIELDS:
            values = [parse_rating(row, field) for row in items]
            finite = [value for value in values if math.isfinite(value)]
            result[f"mean_{field}"] = f"{sum(finite) / len(finite):.6f}" if finite else ""
            result[f"normalized_{field.replace('_1_5', '_0_1')}"] = (
                f"{(sum(finite) / len(finite) - 1.0) / 4.0:.6f}" if finite else ""
            )
        output_rows.append(result)
    if not output_rows:
        raise ValueError("No completed ratings found; set rater_id and at least one 1-5 rating")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {args.output} ({len(output_rows)} clips)")


if __name__ == "__main__":
    main()
