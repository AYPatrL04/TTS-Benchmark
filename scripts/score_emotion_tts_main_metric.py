from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PROSODY_TARGETS = {
    "happy": (0.85, 0.35),
    "angry": (0.90, 0.35),
    "sad": (0.60, 0.40),
    "neutral": (0.80, 0.40),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score emotion-aware TTS samples with a 0-1 composite main metric.")
    parser.add_argument("--input", required=True, type=Path, help="Combined metrics CSV.")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--experiment-name", default="emotion_tts")
    return parser.parse_args()


def parse_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def prosody_fit(target_emotion: str, prosody_activity: float) -> float:
    target, tolerance = PROSODY_TARGETS.get(target_emotion.strip().lower(), PROSODY_TARGETS["neutral"])
    return clamp(1.0 - abs(prosody_activity - target) / tolerance)


def score_row(row: dict[str, str]) -> dict[str, str]:
    wer = clamp(parse_float(row, "wer"), 0.0, 1.0)
    cer = clamp(parse_float(row, "cer"), 0.0, 1.0)
    naturalness_1_5 = parse_float(row, "naturalness_proxy_1_5")
    target_prob = clamp(parse_float(row, "target_emotion_prob"), 0.0, 1.0)
    target_match = clamp(parse_float(row, "target_emotion_match"), 0.0, 1.0)
    prosody_activity = clamp(parse_float(row, "prosody_activity_0_1"), 0.0, 1.0)
    target_emotion = (row.get("target_emotion") or row.get("target_emotion_normalized") or "neutral").strip().lower()

    intelligibility = clamp(0.80 * (1.0 - wer) + 0.20 * (1.0 - cer))
    naturalness = clamp((naturalness_1_5 - 1.0) / 4.0)
    emotion_match = clamp(0.70 * target_prob + 0.30 * target_match)
    prosody = prosody_fit(target_emotion, prosody_activity)

    weighted_raw = clamp(
        0.45 * intelligibility
        + 0.15 * naturalness
        + 0.30 * emotion_match
        + 0.10 * prosody
    )
    intelligibility_gate = clamp(0.35 + 0.65 * intelligibility)
    main_metric = clamp(weighted_raw * intelligibility_gate)

    result = dict(row)
    result.update(
        {
            "main_metric_0_1": f"{main_metric:.6f}",
            "main_metric_raw_0_1": f"{weighted_raw:.6f}",
            "intelligibility_component_0_1": f"{intelligibility:.6f}",
            "naturalness_component_0_1": f"{naturalness:.6f}",
            "emotion_component_0_1": f"{emotion_match:.6f}",
            "prosody_fit_component_0_1": f"{prosody:.6f}",
            "intelligibility_gate_0_1": f"{intelligibility_gate:.6f}",
            "main_metric_formula": (
                "score=(0.45*I+0.15*Q+0.30*E+0.10*P)*(0.35+0.65*I); "
                "I=0.8*(1-WER)+0.2*(1-CER); Q=(naturalness_1_5-1)/4; "
                "E=0.7*target_emotion_prob+0.3*target_emotion_match; "
                "P=target-dependent prosody fit"
            ),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    scored_rows = [score_row(row) for row in rows]
    scored_rows = sorted(scored_rows, key=lambda row: float(row["main_metric_0_1"]), reverse=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in scored_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_rows)

    scores = [float(row["main_metric_0_1"]) for row in scored_rows]
    lines = [
        f"# {args.experiment_name} Composite Main Metric",
        "",
        "All component scores and final scores are normalized to 0-1, higher is better.",
        "",
        "## Formula",
        "",
        "```text",
        "I = 0.80 * (1 - WER) + 0.20 * (1 - CER)",
        "Q = (naturalness_proxy_1_5 - 1) / 4",
        "E = 0.70 * target_emotion_prob + 0.30 * target_emotion_match",
        "P = 1 - abs(prosody_activity - target_prosody) / tolerance",
        "raw = 0.45 * I + 0.15 * Q + 0.30 * E + 0.10 * P",
        "gate = 0.35 + 0.65 * I",
        "main_metric = raw * gate",
        "```",
        "",
        "Prosody targets: "
        + ", ".join(f"{emotion}={target}/tol{tol}" for emotion, (target, tol) in PROSODY_TARGETS.items()),
        "",
        "## Aggregate",
        "",
        f"- input: `{args.input}`",
        f"- samples: {len(scored_rows)}",
        f"- mean main metric: {mean(scores):.6f}",
        f"- best score: {max(scores):.6f}" if scores else "- best score: n/a",
        f"- worst score: {min(scores):.6f}" if scores else "- worst score: n/a",
        "",
        "## Ranked Samples",
        "",
        "| rank | id | target | predicted | score | I | Q | E | P | WER | target_prob | audio |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(scored_rows, start=1):
        audio = row.get("audio_path", "")
        lines.append(
            f"| {rank} | {row.get('id', '')} | {row.get('target_emotion', '')} | "
            f"{row.get('emotion_top_label', '')} | {row['main_metric_0_1']} | "
            f"{row['intelligibility_component_0_1']} | {row['naturalness_component_0_1']} | "
            f"{row['emotion_component_0_1']} | {row['prosody_fit_component_0_1']} | "
            f"{row.get('wer', '')} | {row.get('target_emotion_prob', '')} | `{audio}` |"
        )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
