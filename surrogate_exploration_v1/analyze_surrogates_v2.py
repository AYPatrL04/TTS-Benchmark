from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPLORE_DIR = PROJECT_ROOT / "surrogate_exploration_v1"
OUTPUT_DIR = EXPLORE_DIR / "outputs_v2"
FEATURE_CSV = EXPLORE_DIR / "outputs" / "surrogate_features.csv"

SCORED_DATASETS = [
    (
        "parler_emotion_v1",
        PROJECT_ROOT / "experiments" / "parler_emotion_v1" / "combined" / "parler_emotion_scored_main_metric.csv",
    ),
    (
        "boundary_metric_v1",
        PROJECT_ROOT / "experiments" / "boundary_metric_v1" / "combined" / "boundary_scored_main_metric.csv",
    ),
]

FEATURE_GROUPS = {
    "text_only": ["text_ease"],
    "manifest_text": ["text_ease", "rate_fit", "duration_sec", "speech_rate_wps"],
    "basic_waveform": [
        "signal_quality",
        "rms_dbfs",
        "peak_abs",
        "silence_ratio",
        "spectral_flatness",
        "spectral_centroid_hz",
        "high_freq_ratio",
        "zcr",
    ],
    "prosody_dsp": [
        "prosody_fit_light",
        "prosody_activity_light",
        "f0_std_hz",
        "f0_range_hz",
        "energy_cv",
        "voiced_ratio",
    ],
    "cheap_all": [
        "text_ease",
        "rate_fit",
        "duration_sec",
        "speech_rate_wps",
        "signal_quality",
        "rms_dbfs",
        "peak_abs",
        "silence_ratio",
        "spectral_flatness",
        "spectral_centroid_hz",
        "high_freq_ratio",
        "zcr",
        "prosody_fit_light",
        "prosody_activity_light",
        "f0_std_hz",
        "f0_range_hz",
        "energy_cv",
        "voiced_ratio",
    ],
    "cheap_core4": ["signal_quality", "prosody_fit_light", "rate_fit", "text_ease"],
    "cheap_core3": ["prosody_fit_light", "rate_fit", "text_ease"],
}

MEDIUM_FEATURES = {
    "ser_emotion_component": ["emotion_component_0_1"],
    "ser_plus_dsp": ["emotion_component_0_1", "prosody_fit_light", "rate_fit", "text_ease"],
    "asr_component_only": ["intelligibility_component_0_1"],
    "asr_ser_components": ["intelligibility_component_0_1", "emotion_component_0_1", "quality_component_0_1"],
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def parse_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return default


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = rank
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    import numpy as np

    if len(x) < 2:
        return math.nan
    x_arr = np.asarray(x, dtype="float64")
    y_arr = np.asarray(y, dtype="float64")
    if float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return math.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denom if denom else math.nan


def pairwise_ranking_accuracy(candidate: list[float], target: list[float]) -> float:
    correct = total = 0
    for i in range(len(candidate)):
        for j in range(i + 1, len(candidate)):
            target_delta = target[i] - target[j]
            if target_delta == 0:
                continue
            pred_delta = candidate[i] - candidate[j]
            total += 1
            if pred_delta * target_delta > 0:
                correct += 1
            elif pred_delta == 0:
                correct += 0.5
    return correct / total if total else math.nan


def topk_overlap(candidate: list[float], target: list[float], k: int, largest: bool = True) -> float:
    order_candidate = sorted(range(len(candidate)), key=lambda i: candidate[i], reverse=largest)[:k]
    order_target = sorted(range(len(target)), key=lambda i: target[i], reverse=largest)[:k]
    return len(set(order_candidate) & set(order_target)) / float(k)


def ridge_fit(x, y, alpha: float = 1e-2):
    import numpy as np

    x_arr = np.asarray(x, dtype="float64")
    y_arr = np.asarray(y, dtype="float64")
    mean_x = x_arr.mean(axis=0)
    std_x = x_arr.std(axis=0)
    std_x[std_x == 0] = 1.0
    x_std = (x_arr - mean_x) / std_x
    design = np.column_stack([np.ones(len(x_std)), x_std])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y_arr)
    return coef, mean_x, std_x


def ridge_predict(x, coef, mean_x, std_x) -> list[float]:
    import numpy as np

    x_arr = np.asarray(x, dtype="float64")
    x_std = (x_arr - mean_x) / std_x
    design = np.column_stack([np.ones(len(x_std)), x_std])
    return [clamp(float(value)) for value in design @ coef]


def matrix(rows: list[dict[str, float]], features: list[str]) -> list[list[float]]:
    return [[row[name] for name in features] for row in rows]


def ridge_in_sample(rows: list[dict[str, float]], target: list[float], features: list[str], alpha: float = 1e-2) -> list[float]:
    coef, mean_x, std_x = ridge_fit(matrix(rows, features), target, alpha=alpha)
    return ridge_predict(matrix(rows, features), coef, mean_x, std_x)


def ridge_loo(rows: list[dict[str, float]], target: list[float], features: list[str], alpha: float = 1e-2) -> list[float]:
    preds = []
    for held_out in range(len(rows)):
        train_idx = [idx for idx in range(len(rows)) if idx != held_out]
        train_rows = [rows[idx] for idx in train_idx]
        train_y = [target[idx] for idx in train_idx]
        coef, mean_x, std_x = ridge_fit(matrix(train_rows, features), train_y, alpha=alpha)
        pred = ridge_predict(matrix([rows[held_out]], features), coef, mean_x, std_x)[0]
        preds.append(pred)
    return preds


def ridge_leave_dataset_out(rows: list[dict[str, float]], target: list[float], features: list[str], alpha: float = 1e-2) -> list[float]:
    preds = [math.nan] * len(rows)
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        train_idx = [idx for idx, row in enumerate(rows) if row["dataset"] != dataset]
        test_idx = [idx for idx, row in enumerate(rows) if row["dataset"] == dataset]
        coef, mean_x, std_x = ridge_fit(matrix([rows[idx] for idx in train_idx], features), [target[idx] for idx in train_idx], alpha=alpha)
        test_preds = ridge_predict(matrix([rows[idx] for idx in test_idx], features), coef, mean_x, std_x)
        for idx, pred in zip(test_idx, test_preds):
            preds[idx] = pred
    return preds


def load_scored_components() -> dict[tuple[str, str], dict[str, float]]:
    out = {}
    for dataset_name, path in SCORED_DATASETS:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                out[(dataset_name, row["id"])] = {
                    "intelligibility_component_0_1": parse_float(row, "intelligibility_component_0_1"),
                    "quality_component_0_1": parse_float(row, "quality_component_0_1"),
                    "emotion_component_0_1": parse_float(row, "emotion_component_0_1"),
                }
    return out


def load_rows() -> list[dict[str, float]]:
    component_rows = load_scored_components()
    rows: list[dict[str, float]] = []
    with FEATURE_CSV.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            item: dict[str, float] = {
                "dataset": raw["dataset"],
                "id": raw["id"],
                "case_type": raw.get("case_type", ""),
                "target_emotion": raw.get("target_emotion", ""),
            }
            for key, value in raw.items():
                if key in item:
                    continue
                item[key] = parse_float(raw, key)
            item.update(component_rows[(raw["dataset"], raw["id"])])
            rows.append(item)
    return rows


def evaluate(name: str, preds: list[float], target: list[float], cost_tier: str, ingredients: str, notes: str = "") -> dict[str, str]:
    errors = [abs(pred - label) for pred, label in zip(preds, target)]
    return {
        "candidate": name,
        "cost_tier": cost_tier,
        "ingredients": ingredients,
        "pearson": f"{pearson(preds, target):.6f}",
        "spearman": f"{spearman(preds, target):.6f}",
        "kendall_tau_b": f"{kendall_tau_b(preds, target):.6f}",
        "pairwise_accuracy": f"{pairwise_ranking_accuracy(preds, target):.6f}",
        "mae": f"{mean(errors):.6f}",
        "rmse": f"{math.sqrt(mean(error * error for error in errors)):.6f}",
        "top3_overlap": f"{topk_overlap(preds, target, min(3, len(target))):.6f}",
        "top5_overlap": f"{topk_overlap(preds, target, min(5, len(target))):.6f}",
        "bottom5_overlap": f"{topk_overlap(preds, target, min(5, len(target)), largest=False):.6f}",
        "notes": notes,
    }


def bounded_feature(row: dict[str, float], name: str) -> float:
    value = row[name]
    if name in {"rms_dbfs", "duration_sec", "speech_rate_wps", "spectral_centroid_hz", "f0_std_hz", "f0_range_hz"}:
        return value
    return clamp(value)


def hand_candidates(rows: list[dict[str, float]]) -> dict[str, tuple[list[float], str, str]]:
    out: dict[str, tuple[list[float], str, str]] = {}
    out["text_ease"] = ([row["text_ease"] for row in rows], "very_low", "text difficulty")
    out["rate_fit"] = ([row["rate_fit"] for row in rows], "very_low", "text + duration")
    out["prosody_fit_light"] = ([row["prosody_fit_light"] for row in rows], "low_dsp", "f0 + energy")
    out["signal_quality"] = ([row["signal_quality"] for row in rows], "low_dsp", "RMS + FFT + silence")
    out["cheap_core4_fixed"] = (
        [
            clamp(0.30 * row["signal_quality"] + 0.35 * row["prosody_fit_light"] + 0.20 * row["rate_fit"] + 0.15 * row["text_ease"])
            for row in rows
        ],
        "low_dsp",
        "fixed 4-feature weighted score",
    )
    out["cheap_core3_fixed"] = (
        [clamp(0.45 * row["prosody_fit_light"] + 0.30 * row["rate_fit"] + 0.25 * row["text_ease"]) for row in rows],
        "low_dsp",
        "fixed SER-free style/rate/text score",
    )
    out["ser_emotion_component"] = (
        [row["emotion_component_0_1"] for row in rows],
        "medium_neural",
        "SER emotion component only",
    )
    out["asr_component_only"] = (
        [row["intelligibility_component_0_1"] for row in rows],
        "high_reference",
        "ASR component only",
    )
    return out


def grid_weights_loo(rows: list[dict[str, float]], target: list[float], features: list[str], step: float = 0.10) -> tuple[list[float], tuple[float, ...]]:
    weights = []
    units = int(round(1.0 / step))
    for combo in itertools.product(range(units + 1), repeat=len(features)):
        if sum(combo) != units:
            continue
        weights.append(tuple(value * step for value in combo))

    all_preds = []
    selected_weights = []
    for held_out in range(len(rows)):
        train_idx = [idx for idx in range(len(rows)) if idx != held_out]
        best_weight = weights[0]
        best_score = -math.inf
        for weight in weights:
            train_pred = [
                clamp(sum(weight[i] * rows[idx][features[i]] for i in range(len(features))))
                for idx in train_idx
            ]
            train_target = [target[idx] for idx in train_idx]
            score = spearman(train_pred, train_target)
            if math.isfinite(score) and score > best_score:
                best_score = score
                best_weight = weight
        selected_weights.append(best_weight)
        all_preds.append(clamp(sum(best_weight[i] * rows[held_out][features[i]] for i in range(len(features)))))

    mean_weight = tuple(mean(weight[i] for weight in selected_weights) for i in range(len(features)))
    return all_preds, mean_weight


def exhaustive_subset_search(rows: list[dict[str, float]], target: list[float], features: list[str], max_size: int = 5) -> list[dict[str, str]]:
    results = []
    for size in range(1, max_size + 1):
        for subset in itertools.combinations(features, size):
            preds = ridge_loo(rows, target, list(subset), alpha=5e-2)
            results.append(
                {
                    "features": "+".join(subset),
                    "size": str(size),
                    "pearson": f"{pearson(preds, target):.6f}",
                    "spearman": f"{spearman(preds, target):.6f}",
                    "mae": f"{mean(abs(pred - label) for pred, label in zip(preds, target)):.6f}",
                    "bottom5_overlap": f"{topk_overlap(preds, target, min(5, len(target)), largest=False):.6f}",
                }
            )
    results.sort(key=lambda row: (float(row["spearman"]), float(row["pearson"])), reverse=True)
    return results


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def per_sample_errors(rows: list[dict[str, float]], target: list[float], predictions: dict[str, list[float]]) -> list[dict[str, str]]:
    out = []
    for idx, row in enumerate(rows):
        item = {
            "dataset": row["dataset"],
            "id": row["id"],
            "case_type": row.get("case_type", ""),
            "target_emotion": row.get("target_emotion", ""),
            "main_metric_0_1": f"{target[idx]:.6f}",
        }
        for name, preds in predictions.items():
            item[f"{name}_pred"] = f"{preds[idx]:.6f}"
            item[f"{name}_abs_error"] = f"{abs(preds[idx] - target[idx]):.6f}"
        out.append(item)
    return out


def main() -> None:
    rows = load_rows()
    target = [row["main_metric_0_1"] for row in rows]
    candidate_rows = []
    prediction_map: dict[str, list[float]] = {}

    for name, (preds, cost_tier, ingredients) in hand_candidates(rows).items():
        prediction_map[name] = preds
        candidate_rows.append(evaluate(name, preds, target, cost_tier, ingredients))

    predefined = {
        "ridge_text_only_loo": ("very_low", FEATURE_GROUPS["text_only"]),
        "ridge_manifest_text_loo": ("very_low", FEATURE_GROUPS["manifest_text"]),
        "ridge_basic_waveform_loo": ("low_dsp", FEATURE_GROUPS["basic_waveform"]),
        "ridge_prosody_dsp_loo": ("low_dsp", FEATURE_GROUPS["prosody_dsp"]),
        "ridge_cheap_core3_loo": ("low_dsp", FEATURE_GROUPS["cheap_core3"]),
        "ridge_cheap_core4_loo": ("low_dsp", FEATURE_GROUPS["cheap_core4"]),
        "ridge_cheap_all_loo": ("low_dsp", FEATURE_GROUPS["cheap_all"]),
        "ridge_ser_plus_dsp_loo": ("medium_neural", MEDIUM_FEATURES["ser_plus_dsp"]),
        "ridge_asr_ser_components_loo": ("high_reference", MEDIUM_FEATURES["asr_ser_components"]),
    }
    for name, (cost_tier, features) in predefined.items():
        preds = ridge_loo(rows, target, features, alpha=5e-2)
        prediction_map[name] = preds
        candidate_rows.append(evaluate(name, preds, target, cost_tier, "+".join(features), "LOOCV ridge"))

    for name, features in {
        "ridge_cheap_core4_leave_dataset_out": FEATURE_GROUPS["cheap_core4"],
        "ridge_prosody_dsp_leave_dataset_out": FEATURE_GROUPS["prosody_dsp"],
        "ridge_ser_plus_dsp_leave_dataset_out": MEDIUM_FEATURES["ser_plus_dsp"],
    }.items():
        preds = ridge_leave_dataset_out(rows, target, features, alpha=5e-2)
        prediction_map[name] = preds
        cost_tier = "medium_neural" if "ser" in name else "low_dsp"
        candidate_rows.append(evaluate(name, preds, target, cost_tier, "+".join(features), "leave-one-dataset-out"))

    grid_features = ["signal_quality", "prosody_fit_light", "rate_fit", "text_ease"]
    grid_preds, mean_weight = grid_weights_loo(rows, target, grid_features, step=0.10)
    prediction_map["grid_weighted_core4_loo"] = grid_preds
    candidate_rows.append(
        evaluate(
            "grid_weighted_core4_loo",
            grid_preds,
            target,
            "low_dsp",
            "+".join(grid_features),
            "LOOCV grid weights mean=" + ",".join(f"{weight:.3f}" for weight in mean_weight),
        )
    )

    subset_features = [
        "text_ease",
        "rate_fit",
        "duration_sec",
        "speech_rate_wps",
        "silence_ratio",
        "energy_cv",
        "f0_std_hz",
        "f0_range_hz",
        "voiced_ratio",
        "zcr",
        "spectral_flatness",
        "spectral_centroid_hz",
        "high_freq_ratio",
        "prosody_fit_light",
        "prosody_activity_light",
    ]
    subset_rows = exhaustive_subset_search(rows, target, subset_features, max_size=4)
    best_subset_features = subset_rows[0]["features"].split("+")
    best_subset_preds = ridge_loo(rows, target, best_subset_features, alpha=5e-2)
    prediction_map["best_subset_ridge_loo"] = best_subset_preds
    candidate_rows.append(
        evaluate(
            "best_subset_ridge_loo",
            best_subset_preds,
            target,
            "low_dsp",
            "+".join(best_subset_features),
            "best exhaustive subset up to 4 features; selection is in-sample and optimistic",
        )
    )

    candidate_rows.sort(key=lambda row: (float(row["spearman"]), float(row["pearson"])), reverse=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "surrogate_candidates_v2.csv", candidate_rows)
    write_csv(OUTPUT_DIR / "subset_search_top30.csv", subset_rows[:30])
    write_csv(
        OUTPUT_DIR / "surrogate_error_analysis.csv",
        per_sample_errors(
            rows,
            target,
            {
                "ridge_cheap_core4_loo": prediction_map["ridge_cheap_core4_loo"],
                "best_subset_ridge_loo": prediction_map["best_subset_ridge_loo"],
                "ridge_ser_plus_dsp_loo": prediction_map["ridge_ser_plus_dsp_loo"],
                "grid_weighted_core4_loo": prediction_map["grid_weighted_core4_loo"],
            },
        ),
    )

    top_candidates = candidate_rows[:12]
    lines = [
        "# Surrogate Metric Exploration V2",
        "",
        "This report compares several low-compute surrogate strategies against the current `main_metric_0_1`.",
        "",
        "## Cost Tiers",
        "",
        "| tier | approximate cost | examples |",
        "| --- | --- | --- |",
        "| `very_low` | text parsing plus manifest duration; no waveform FFT, no neural model | `text_ease`, `rate_fit` |",
        "| `low_dsp` | CPU waveform features such as RMS, FFT, f0 autocorrelation; no ASR/SER | `prosody_fit_light`, ridge cheap features |",
        "| `medium_neural` | uses an emotion classifier output; cheaper than full ASR+SER main metric but not a truly cheap surrogate | SER + DSP candidates |",
        "| `high_reference` | uses ASR/SER components from the main metric; included only as an upper-bound sanity check | ASR/SER components |",
        "",
        "## Top Results",
        "",
        "| candidate | tier | Pearson | Spearman | MAE | top3 | top5 | bottom5 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_candidates:
        lines.append(
            f"| {row['candidate']} | {row['cost_tier']} | {row['pearson']} | {row['spearman']} | {row['mae']} | {row['top3_overlap']} | {row['top5_overlap']} | {row['bottom5_overlap']} |"
        )
    best_low = next(row for row in candidate_rows if row["cost_tier"] == "low_dsp")
    best_very_low = next(row for row in candidate_rows if row["cost_tier"] == "very_low")
    best_medium = next(row for row in candidate_rows if row["cost_tier"] == "medium_neural")
    lines.extend(
        [
            "",
            "## Recommended Candidates",
            "",
            f"- Best practical low-DSP candidate: `{best_low['candidate']}` with Pearson `{best_low['pearson']}`, Spearman `{best_low['spearman']}`, MAE `{best_low['mae']}`.",
            f"- Best very-low-cost candidate: `{best_very_low['candidate']}` with Pearson `{best_very_low['pearson']}`, Spearman `{best_very_low['spearman']}`, MAE `{best_very_low['mae']}`.",
            f"- Best medium-neural upper-bound candidate: `{best_medium['candidate']}` with Pearson `{best_medium['pearson']}`, Spearman `{best_medium['spearman']}`, MAE `{best_medium['mae']}`.",
            "",
            "## Interpretation",
            "",
            "- The low-DSP candidates can partially track the main metric, but they are not close to the `~0.930` SIM/WER correlation reported by Yufan.",
            "- The major missing signal is emotion classification: cheap DSP prosody can approximate style movement, but it cannot reliably detect whether SER would call the audio happy, sad, angry, or neutral.",
            "- The current audio quality features are weak in this dataset because most generated clips are clean; they add little ranking power.",
            "- Leave-one-dataset-out scores are much worse than LOOCV for cheap candidates, so the current sample set is too small and distribution-specific for deployment.",
            "- SER-based medium candidates show the ceiling improves when emotion information is available, but that reintroduces neural compute.",
            "",
            "## Files",
            "",
            "- `outputs_v2/surrogate_candidates_v2.csv`",
            "- `outputs_v2/subset_search_top30.csv`",
            "- `outputs_v2/surrogate_error_analysis.csv`",
            "- `outputs_v2/surrogate_report_v2.md`",
        ]
    )
    (OUTPUT_DIR / "surrogate_report_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_DIR / 'surrogate_candidates_v2.csv'}")
    print(f"wrote {OUTPUT_DIR / 'subset_search_top30.csv'}")
    print(f"wrote {OUTPUT_DIR / 'surrogate_error_analysis.csv'}")
    print(f"wrote {OUTPUT_DIR / 'surrogate_report_v2.md'}")


if __name__ == "__main__":
    main()
