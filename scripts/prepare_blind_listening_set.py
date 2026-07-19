from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a randomized blind listening set and rating template.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_audio(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    rng = random.Random(args.seed)
    rng.shuffle(source_rows)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    public_rows = []
    private_rows = []
    for index, row in enumerate(source_rows, start=1):
        blind_id = f"clip_{index:03d}"
        source = resolve_audio(row["audio_path"])
        destination = audio_dir / f"{blind_id}{source.suffix.lower()}"
        if args.overwrite or not destination.exists():
            shutil.copy2(source, destination)
        public_rows.append(
            {
                "rater_id": "",
                "blind_id": blind_id,
                "text": row["text"],
                "target_emotion": row.get("target_emotion", ""),
                "is_boundary": row.get("is_boundary", ""),
                "boundary_type": row.get("boundary_type", ""),
                "intelligibility_1_5": "",
                "naturalness_1_5": "",
                "emotion_match_1_5": "",
                "overall_acceptability_1_5": "",
                "heard_transcript_optional": "",
                "notes_optional": "",
            }
        )
        private = {"blind_id": blind_id, **row}
        private_rows.append(private)

    write_csv(args.output_dir / "ratings_template.csv", public_rows)
    write_csv(args.output_dir / "private_blind_key.csv", private_rows)
    instructions = """# Blind Listening Instructions

Listen to files in `audio/` without opening `private_blind_key.csv`.

Use one row per rater and clip. For additional raters, duplicate the 18 template rows and set a different anonymous `rater_id`.

- `intelligibility_1_5`: 1 = impossible to recover the sentence; 5 = every intended word is clear.
- `naturalness_1_5`: 1 = unusable or severely synthetic; 5 = natural human-like speech without disturbing artifacts.
- `emotion_match_1_5`: 1 = strongly conflicts with the requested emotion; 5 = clearly matches it. The current set requests neutral delivery.
- `overall_acceptability_1_5`: 1 = reject; 5 = fully acceptable for the intended use.
- `heard_transcript_optional`: recommended for boundary clips; type what you heard without looking up model identity.

Use at least three raters for a pilot and five or more for metric calibration. Do not reveal model, voice, Main score, or surrogate score before ratings are complete. Keep `private_blind_key.csv` separate until all ratings are locked.
"""
    (args.output_dir / "README.md").write_text(instructions, encoding="utf-8")
    print(f"wrote blind set to {args.output_dir}")


if __name__ == "__main__":
    main()
