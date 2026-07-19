from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join the three main metric tracks into one report.")
    parser.add_argument("--intelligibility", required=True, type=Path)
    parser.add_argument("--naturalness", required=True, type=Path)
    parser.add_argument("--style", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def read_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return math.nan


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def main() -> None:
    args = parse_args()
    intelligibility = read_by_id(args.intelligibility)
    naturalness = read_by_id(args.naturalness)
    style = read_by_id(args.style)
    ids = [item_id for item_id in intelligibility.keys() if item_id in naturalness and item_id in style]

    rows = []
    for item_id in ids:
        irow = intelligibility[item_id]
        nrow = naturalness[item_id]
        srow = style[item_id]
        wer = f(irow, "wer")
        acoustic_sanity = f(nrow, "acoustic_sanity_score_0_1")
        if not math.isfinite(acoustic_sanity):
            legacy_naturalness = f(nrow, "naturalness_proxy_1_5")
            acoustic_sanity = (legacy_naturalness - 1.0) / 4.0 if math.isfinite(legacy_naturalness) else math.nan
        prosody = f(srow, "prosody_activity_0_1")
        emotion_top_prob = f(srow, "emotion_top_prob")
        target_emotion_prob = f(srow, "target_emotion_prob")
        intelligibility_score_0_100 = max(0.0, 1.0 - min(wer, 1.0)) * 100.0 if math.isfinite(wer) else math.nan
        style_emotion_prob = target_emotion_prob if math.isfinite(target_emotion_prob) else emotion_top_prob
        style_proxy_0_100 = (
            0.5 * prosody * 100.0 + 0.5 * style_emotion_prob * 100.0
            if math.isfinite(prosody) and math.isfinite(style_emotion_prob)
            else math.nan
        )
        metadata_fields = (
            "text_id",
            "tts_system",
            "model",
            "voice",
            "source_config",
            "is_boundary",
            "boundary_type",
            "description",
        )
        result = {
                "id": item_id,
                "case_type": irow.get("case_type", ""),
                "expected_metric_challenge": irow.get("expected_metric_challenge", ""),
                "target_emotion": irow.get("target_emotion", "") or srow.get("target_emotion", ""),
                "text": irow.get("text", ""),
                "audio_path": irow.get("audio_path") or irow.get("generated_audio", ""),
                "wer": irow.get("wer", ""),
                "cer": irow.get("cer", ""),
                "intelligibility_score_0_100": f"{intelligibility_score_0_100:.3f}",
                "acoustic_sanity_score_0_1": f"{acoustic_sanity:.6f}" if math.isfinite(acoustic_sanity) else "",
                "emotion_top_label": srow.get("emotion_top_label", ""),
                "emotion_top_prob": srow.get("emotion_top_prob", ""),
                "target_emotion_normalized": srow.get("target_emotion_normalized", ""),
                "target_emotion_prob": srow.get("target_emotion_prob", ""),
                "target_emotion_match": srow.get("target_emotion_match", ""),
                "prosody_activity_0_1": srow.get("prosody_activity_0_1", ""),
                "f0_median_hz": srow.get("f0_median_hz", ""),
                "f0_std_semitones": srow.get("f0_std_semitones", ""),
                "f0_mad_semitones": srow.get("f0_mad_semitones", ""),
                "f0_range_semitones_p90_p10": srow.get("f0_range_semitones_p90_p10", ""),
                "energy_cv": srow.get("energy_cv", ""),
                "voiced_ratio": srow.get("voiced_ratio", ""),
                "style_proxy_0_100": f"{style_proxy_0_100:.3f}",
                "coarse_codec_sim_like": irow.get("coarse_codec_sim_like", ""),
                "coarse_codec_nll": irow.get("coarse_codec_nll", ""),
                "silence_ratio": irow.get("silence_ratio", ""),
                "asr_transcript": irow.get("asr_transcript", ""),
            }
        for field in metadata_fields:
            result[field] = irow.get(field, "") or srow.get(field, "") or nrow.get(field, "")
        rows.append(result)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    worst_wer = sorted(rows, key=lambda row: f(row, "wer"), reverse=True)[:5]
    lowest_naturalness = sorted(rows, key=lambda row: f(row, "acoustic_sanity_score_0_1"))[:5]
    lowest_style = sorted(rows, key=lambda row: f(row, "style_proxy_0_100"))[:5]
    emotion_counts: dict[str, int] = {}
    for row in rows:
        emotion = row["emotion_top_label"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    lines = [
        "# Main Metric Components V2 Report",
        "",
        "This report joins three automatic main metric tracks:",
        "",
        "- Intelligibility: Whisper-normalized ASR WER/CER.",
        "- Acoustic sanity: clipping, silence, loudness, duration, and flatness checks. This is not naturalness or MOS.",
        "- Style/emotion: SUPERB Wav2Vec2 emotion classifier plus pitch/energy prosody activity.",
        "",
        "## Aggregate",
        "",
        f"- samples: {len(rows)}",
        f"- mean WER: {mean([f(row, 'wer') for row in rows]):.6f}",
        f"- mean CER: {mean([f(row, 'cer') for row in rows]):.6f}",
        f"- mean intelligibility score: {mean([f(row, 'intelligibility_score_0_100') for row in rows]):.3f}/100",
        f"- mean acoustic sanity: {mean([f(row, 'acoustic_sanity_score_0_1') for row in rows]):.6f}",
        f"- mean prosody activity: {mean([f(row, 'prosody_activity_0_1') for row in rows]):.6f}",
        f"- emotion top-label counts: {emotion_counts}",
        "",
        "## Worst WER",
        "",
        "| id | WER | CER | naturalness | emotion | prosody | transcript |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in worst_wer:
        transcript = row["asr_transcript"].replace("|", "\\|")
        lines.append(
            f"| {row['id']} | {row['wer']} | {row['cer']} | {row['acoustic_sanity_score_0_1']} | {row['emotion_top_label']} | {row['prosody_activity_0_1']} | {transcript} |"
        )

    lines.extend(["", "## Lowest Acoustic Sanity", "", "| id | acoustic sanity | WER | silence_ratio | audio |", "| --- | ---: | ---: | ---: | --- |"])
    for row in lowest_naturalness:
        lines.append(
            f"| {row['id']} | {row['acoustic_sanity_score_0_1']} | {row['wer']} | {row['silence_ratio']} | `{row['audio_path']}` |"
        )

    lines.extend(["", "## Lowest Style Proxy", "", "| id | style_proxy | emotion | emotion_prob | prosody | audio |", "| --- | ---: | --- | ---: | ---: | --- |"])
    for row in lowest_style:
        lines.append(
            f"| {row['id']} | {row['style_proxy_0_100']} | {row['emotion_top_label']} | {row['emotion_top_prob']} | {row['prosody_activity_0_1']} | `{row['audio_path']}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- WER behaves as an intelligibility metric: substitutions, deletions, and insertions show where ASR could not recover the intended text.",
            "- Acoustic sanity catches gross failures only. Add calibrated UTMOS and defect dimensions before interpreting Q as naturalness.",
            "- Style/emotion results should be read as an automatic screening signal; SER can be sensitive to speaker identity, wording, and prosody artifacts.",
            "- These automatic main metrics are suitable for screening and surrogate fitting, but still need a small human listening sanity check for boundary cases.",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
