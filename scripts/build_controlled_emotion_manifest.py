from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "automatic_emotion_consensus_v1" / "controlled_generation"


def read_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def main() -> None:
    generated = read_index(BASE / "generated_manifest.csv")
    wer = read_index(BASE / "wer.csv")
    sanity = read_index(BASE / "sanity.csv")
    prosody = read_index(BASE / "legacy_ser_prosody.csv")
    rows = []
    for item_id, row in generated.items():
        merged = dict(row)
        for source in (wer[item_id], sanity[item_id], prosody[item_id]):
            merged.update(source)
        merged.update(
            {
                "sample_key": f"controlled_emotion_intensity_v1::{item_id}",
                "dataset": "controlled_emotion_intensity_v1",
                "tts_system": "parler",
                "voice": "Jenna",
                "case_type": "controlled_same_text",
                "is_boundary": "0",
            }
        )
        rows.append(merged)

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    output = BASE / "evaluation_manifest.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} ({len(rows)} clips)")


if __name__ == "__main__":
    main()
