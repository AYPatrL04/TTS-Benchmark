from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPLORE_DIR = PROJECT_ROOT / "surrogate_exploration_v1"
OUTPUT_DIR = EXPLORE_DIR / "outputs_v3" / "cost_measurement"
COMBINED_INPUT = OUTPUT_DIR / "combined_26_input.csv"

DATASETS = [
    PROJECT_ROOT / "experiments" / "parler_emotion_v1" / "combined" / "parler_emotion_scored_main_metric.csv",
    PROJECT_ROOT / "experiments" / "boundary_metric_v1" / "combined" / "boundary_scored_main_metric.csv",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def combine_inputs() -> list[dict[str, str]]:
    rows = []
    for path in DATASETS:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows.extend(list(csv.DictReader(handle)))
    write_csv(COMBINED_INPUT, rows)
    return rows


def measure_callable(name: str, sample_count: int, func) -> dict[str, object]:
    start = time.perf_counter()
    func()
    seconds = time.perf_counter() - start
    return {
        "name": name,
        "status": "success",
        "seconds_total": f"{seconds:.6f}",
        "seconds_per_clip": f"{seconds / sample_count:.6f}",
        "notes": "in-process timing",
    }


def measure_command(name: str, sample_count: int, args: list[str]) -> dict[str, object]:
    start = time.perf_counter()
    completed = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
    )
    seconds = time.perf_counter() - start
    log_path = OUTPUT_DIR / f"{name}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "name": name,
        "status": "success" if completed.returncode == 0 else f"failed:{completed.returncode}",
        "seconds_total": f"{seconds:.6f}",
        "seconds_per_clip": f"{seconds / sample_count:.6f}",
        "notes": f"subprocess timing; log={log_path.name}",
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = combine_inputs()
    sample_count = len(rows)

    sys.path.insert(0, str(EXPLORE_DIR))
    import analyze_surrogates
    import analyze_surrogates_v3

    measurements = []

    def very_low_once() -> None:
        for row in rows:
            text_features = analyze_surrogates.text_difficulty(row.get("text", ""))
            duration = float(row.get("audio_duration_sec") or row.get("nat_duration_sec") or 0.0)
            _ = text_features["text_ease"]
            _ = text_features["word_count"] / max(duration, 1e-8)

    def very_low_repeated() -> None:
        for _ in range(200):
            very_low_once()

    very_low = measure_callable("very_low_text_duration_x200", sample_count * 200, very_low_repeated)
    very_low["equivalent_26_clip_seconds"] = f"{float(very_low['seconds_per_clip']) * sample_count:.6f}"
    measurements.append(very_low)

    def base_low_dsp_once() -> None:
        for row in analyze_surrogates.read_rows():
            _ = analyze_surrogates.text_difficulty(row.get("text", ""))
            _ = analyze_surrogates.acoustic_features(row)

    measurements.append(measure_callable("low_dsp_base_features", sample_count, base_low_dsp_once))

    def enhanced_low_dsp_once() -> None:
        _rows, _elapsed = analyze_surrogates_v3.load_rows_with_audio_features()

    measurements.append(measure_callable("low_dsp_enhanced_features_v3", sample_count, enhanced_low_dsp_once))

    def fixed_formula_repeated() -> None:
        feature_rows, _elapsed = analyze_surrogates_v3.load_rows_with_audio_features()
        for _ in range(1000):
            for row in feature_rows:
                delivery = analyze_surrogates_v3.clamp(
                    0.36 * row["target_style_fit_v1"]
                    + 0.22 * row["prosody_fit_light"]
                    + 0.16 * row["pause_naturalness"]
                    + 0.14 * row["spectral_balance_fit"]
                    + 0.12 * row["articulation_risk_inverse"]
                )
                _ = analyze_surrogates_v3.clamp(
                    0.38 * delivery
                    + 0.18 * row["text_ease"]
                    + 0.16 * row["rate_fit"]
                    + 0.14 * row["voice_presence_fit"]
                    + 0.14 * row["signal_quality"]
                )

    formula = measure_callable("fixed_surrogate_formula_x1000", sample_count * 1000, fixed_formula_repeated)
    formula["equivalent_26_clip_seconds"] = f"{float(formula['seconds_per_clip']) * sample_count:.6f}"
    measurements.append(formula)

    py = sys.executable
    measurements.append(
        measure_command(
            "main_naturalness_proxy",
            sample_count,
            [
                py,
                "scripts/evaluate_acoustic_naturalness_proxy.py",
                "--input",
                str(COMBINED_INPUT),
                "--output-csv",
                str(OUTPUT_DIR / "naturalness_proxy.csv"),
                "--output-md",
                str(OUTPUT_DIR / "naturalness_proxy.md"),
            ],
        )
    )
    measurements.append(
        measure_command(
            "main_emotion_ser_prosody",
            sample_count,
            [
                py,
                "scripts/evaluate_emotion_prosody.py",
                "--input",
                str(COMBINED_INPUT),
                "--output-csv",
                str(OUTPUT_DIR / "emotion_prosody.csv"),
                "--output-md",
                str(OUTPUT_DIR / "emotion_prosody.md"),
                "--device",
                "auto",
            ],
        )
    )
    measurements.append(
        measure_command(
            "main_whisper_wer",
            sample_count,
            [
                py,
                "scripts/evaluate_wer_with_transformers_whisper.py",
                "--input",
                str(COMBINED_INPUT),
                "--output-csv",
                str(OUTPUT_DIR / "asr_wer.csv"),
                "--output-md",
                str(OUTPUT_DIR / "asr_wer.md"),
                "--device",
                "auto",
                "--overwrite",
            ],
        )
    )

    def score_only_once() -> None:
        args = [
            py,
            "scripts/score_emotion_tts_main_metric.py",
            "--input",
            str(COMBINED_INPUT),
            "--output-csv",
            str(OUTPUT_DIR / "main_score_only.csv"),
            "--output-md",
            str(OUTPUT_DIR / "main_score_only.md"),
            "--experiment-name",
            "timing_26",
        ]
        completed = subprocess.run(args, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (OUTPUT_DIR / "main_score_only.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout)

    measurements.append(measure_callable("main_composite_score_only", sample_count, score_only_once))

    success_measurements = [row for row in measurements if str(row["status"]) == "success"]
    main_component_names = {
        "main_naturalness_proxy",
        "main_emotion_ser_prosody",
        "main_whisper_wer",
        "main_composite_score_only",
    }
    main_total = sum(float(row["seconds_total"]) for row in success_measurements if row["name"] in main_component_names)
    low_dsp_total = sum(
        float(row["seconds_total"])
        for row in success_measurements
        if row["name"] in {"low_dsp_base_features", "low_dsp_enhanced_features_v3"}
    )
    very_low_total = float(very_low["equivalent_26_clip_seconds"])
    formula_total = float(formula["equivalent_26_clip_seconds"])

    summary = [
        {
            "scenario": "main_metric_current_pipeline",
            "seconds_total_26": f"{main_total:.6f}",
            "seconds_per_clip": f"{main_total / sample_count:.6f}",
            "relative_to_main": "1.000000",
        },
        {
            "scenario": "very_low_text_duration",
            "seconds_total_26": f"{very_low_total:.6f}",
            "seconds_per_clip": f"{very_low_total / sample_count:.6f}",
            "relative_to_main": f"{very_low_total / main_total:.8f}" if main_total else "",
        },
        {
            "scenario": "low_dsp_base_plus_v3_features",
            "seconds_total_26": f"{low_dsp_total:.6f}",
            "seconds_per_clip": f"{low_dsp_total / sample_count:.6f}",
            "relative_to_main": f"{low_dsp_total / main_total:.8f}" if main_total else "",
        },
        {
            "scenario": "fixed_surrogate_formula_only",
            "seconds_total_26": f"{formula_total:.6f}",
            "seconds_per_clip": f"{formula_total / sample_count:.6f}",
            "relative_to_main": f"{formula_total / main_total:.8f}" if main_total else "",
        },
    ]

    write_csv(OUTPUT_DIR / "metric_cost_measurements.csv", measurements)
    write_csv(OUTPUT_DIR / "metric_cost_summary.csv", summary)

    lines = [
        "# Metric Cost Measurement",
        "",
        f"Samples: {sample_count}",
        "",
        "## Summary",
        "",
        "| scenario | seconds total / 26 clips | seconds / clip | relative to main |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| `{row['scenario']}` | {row['seconds_total_26']} | {row['seconds_per_clip']} | {row['relative_to_main']} |"
        )
    lines.extend(
        [
            "",
            "## Components",
            "",
            "| name | status | seconds total | seconds / clip | notes |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in measurements:
        lines.append(
            f"| `{row['name']}` | {row['status']} | {row['seconds_total']} | {row['seconds_per_clip']} | {row['notes']} |"
        )
    (OUTPUT_DIR / "metric_cost_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_DIR / 'metric_cost_summary.csv'}")
    print(f"wrote {OUTPUT_DIR / 'metric_cost_measurements.csv'}")
    print(f"wrote {OUTPUT_DIR / 'metric_cost_report.md'}")


if __name__ == "__main__":
    main()
