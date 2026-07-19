from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "automatic_emotion_consensus_v1"
SOURCES = (
    BASE / "model_outputs" / "emotion_model_outputs.csv",
    BASE / "controlled_generation" / "emotion_model_outputs.csv",
)
OUTPUT = BASE / "model_outputs" / "emotion_model_outputs_all_52.csv"


def main() -> None:
    rows = []
    fields: list[str] = []
    for path in SOURCES:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
                for key in row:
                    if key not in fields:
                        fields.append(key)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUT} ({len(rows)} clips)")


if __name__ == "__main__":
    main()
