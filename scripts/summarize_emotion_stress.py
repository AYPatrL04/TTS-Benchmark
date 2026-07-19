from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize main metrics by target emotion.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def fmt(value: float) -> str:
    return f"{value:.6f}" if math.isfinite(value) else "nan"


def main() -> None:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        target = row.get("target_emotion") or row.get("target_emotion_normalized") or "unknown"
        groups[target].append(row)

    lines = [
        "# Emotion Stress Main Metrics",
        "",
        f"- input: `{args.input}`",
        f"- samples: {len(rows)}",
        "",
        "This summarizes the three automatic main metric tracks by intended text emotion.",
        "",
        "## By Target Emotion",
        "",
        "| target | n | mean WER | mean naturalness | mean target emotion prob | match rate | mean prosody |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for target in sorted(groups):
        items = groups[target]
        mean_wer = mean([parse_float(row.get("wer")) for row in items])
        mean_nat = mean([parse_float(row.get("naturalness_proxy_1_5")) for row in items])
        mean_target_prob = mean([parse_float(row.get("target_emotion_prob")) for row in items])
        mean_match = mean([parse_float(row.get("target_emotion_match")) for row in items])
        mean_prosody = mean([parse_float(row.get("prosody_activity_0_1")) for row in items])
        lines.append(
            f"| {target} | {len(items)} | {fmt(mean_wer)} | {fmt(mean_nat)} | {fmt(mean_target_prob)} | {fmt(mean_match)} | {fmt(mean_prosody)} |"
        )

    lines.extend(
        [
            "",
            "## Per Sample",
            "",
            "| id | target | WER | naturalness | predicted emotion | target prob | match | prosody | transcript |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        transcript = (row.get("asr_transcript") or "").replace("|", "\\|")
        target = row.get("target_emotion") or row.get("target_emotion_normalized") or ""
        lines.append(
            "| {id} | {target} | {wer} | {nat} | {pred} | {target_prob} | {match} | {prosody} | {transcript} |".format(
                id=row.get("id", ""),
                target=target,
                wer=row.get("wer", ""),
                nat=row.get("naturalness_proxy_1_5", ""),
                pred=row.get("emotion_top_label", ""),
                target_prob=row.get("target_emotion_prob", ""),
                match=row.get("target_emotion_match", ""),
                prosody=row.get("prosody_activity_0_1", ""),
                transcript=transcript,
            )
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- WER/CER measure text intelligibility and should be interpreted separately from style or emotion control.",
            "- Target emotion probability and match rate are automatic classifier checks, not human ground truth.",
            "- High prosody activity means the audio has pitch/energy variation; it does not prove that the intended emotion was expressed.",
            "- A flat naturalness proxy means this fallback metric is mostly catching gross acoustic defects, not subtle emotional quality.",
            "- Low style rows should be prioritized for listening checks because automatic SER models can confuse affect, speaker traits, and lexical content.",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
