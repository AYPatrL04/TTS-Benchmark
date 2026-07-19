# Main and Surrogate Metric Comparison

## Scope

This experiment uses 52 automatically evaluated English TTS clips. It contains
Parler-TTS, Bark, and Windows SAPI audio; ordinary and boundary cases; and eight
Parler clips that use exactly the same text and speaker while varying only the
emotion and intensity description. No human rating is used.

The generation prompt label is treated as the requested emotion. It is not a
human-confirmed perceptual label. The Main metric is therefore an automatic
teacher for pipeline experiments, not perceptual ground truth.

## Main Metric

### Intelligibility

Whisper `tiny.en` produces the transcript used for normalized WER:

```text
WER = (substitutions + deletions + insertions) / reference_word_count
I   = clip(1 - WER, 0, 1)
```

CER and the transcript remain diagnostics. They do not provide independent
evidence because they come from the same ASR output.

### Automatic Emotion Consensus

Three heterogeneous signals are evaluated:

1. `emotion2vec/emotion2vec_plus_base` target-class probability.
2. `superb/wav2vec2-base-superb-er` target-class probability.
3. `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` continuous
   arousal, dominance, and valence aligned to fixed, non-fitted emotion anchors.

For VAD point `v=(a,d,val)`, anchor `c_e`, and scale
`s=(0.25,0.25,0.30)`:

```text
logit_vad(e) = -0.5 * sum_k ((v_k - c_e,k) / s_k)^2
P_vad(e)     = softmax(logit_vad)(e)

c_neutral = (0.40, 0.50, 0.50)
c_happy   = (0.70, 0.70, 0.75)
c_angry   = (0.75, 0.70, 0.25)
c_sad     = (0.30, 0.35, 0.30)
```

The robust emotion component and disagreement diagnostic are:

```text
E = median(P_e2v(target), P_SUPERB(target), P_vad(target))
D = max(P_e2v, P_SUPERB, P_vad) - min(P_e2v, P_SUPERB, P_vad)
```

The median prevents one failed classifier from controlling the score. `D` is
not evidence of bad speech by itself, but high `D` means the emotion result is
not reliable enough for a strong claim.

### Acoustic Sanity

The existing sanity detector checks gross failures using loudness, silence,
clipping, spectral flatness, and short-duration penalties:

```text
S = clip(1 - (0.30*loudness_penalty
            + 0.30*silence_penalty
            + 0.20*clipping_penalty
            + 0.15*flatness_penalty
            + 0.05*short_duration_penalty), 0, 1)
```

This is not naturalness MOS. It does not reliably detect robotic timbre,
vocoder artifacts, reverberation, or unnatural phoneme timing.

### Composite

The automatic Main score is a weighted geometric mean:

```text
Main_auto_v3 = I^0.55 * E^0.35 * S^0.10
eligible     = (I >= 0.70) and (S >= 0.50)
```

Unlike a weighted arithmetic sum, the geometric form does not allow excellent
emotion evidence to fully compensate for unintelligible speech, or vice versa.
The exponents are design choices and have not been calibrated against people.

## Controlled Emotion Result

The table below holds text, speaker, TTS model, seed, and decoding settings
constant. Only the style description changes.

| sample | e2v target P | SUPERB target P | VAD target P | A | D | V | prosody | E | Main |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| neutral subtle | 0.955 | 0.825 | 0.517 | 0.382 | 0.487 | 0.612 | 0.752 | 0.825 | 0.898 |
| neutral obvious | 1.000 | 0.724 | 0.535 | 0.301 | 0.431 | 0.639 | 0.685 | 0.724 | 0.742 |
| happy subtle | 0.000 | 0.556 | 0.238 | 0.427 | 0.527 | 0.659 | 0.748 | 0.238 | 0.530 |
| happy obvious | 0.000 | 0.691 | 0.326 | 0.477 | 0.554 | 0.693 | 0.774 | 0.326 | 0.620 |
| sad subtle | 1.000 | 0.015 | 0.283 | 0.363 | 0.482 | 0.603 | 0.722 | 0.283 | 0.563 |
| sad obvious | 1.000 | 0.006 | 0.228 | 0.384 | 0.481 | 0.692 | 0.735 | 0.228 | 0.572 |
| angry subtle | 0.000 | 0.006 | 0.072 | 0.408 | 0.515 | 0.640 | 0.720 | 0.006 | 0.158 |
| angry obvious | 0.000 | 0.002 | 0.071 | 0.448 | 0.544 | 0.730 | 0.800 | 0.002 | 0.118 |

The model produces a detectable happy intensity change: obvious happy raises
SUPERB target probability, VAD alignment, and prosody activity. The angry
prompt increases activity but also raises valence, so the automatic models do
not interpret it as anger. Sad has strong emotion2vec evidence but conflicts
with SUPERB and VAD. Parler therefore changes intonation, but does not reliably
deliver all requested emotions in this controlled configuration.

## Surrogate Metrics

Each learned surrogate uses ridge regression with fold-local standardization
and fixed regularization:

```text
z_j   = (x_j - mean_train_j) / std_train_j
beta* = argmin sum_i (Main_i - beta_0 - z_i beta)^2 + 10*||beta||_2^2
Surr  = clip(beta_0 + z beta, 0, 1)
```

Candidate feature sets are:

```text
low-DSP = signal quality, rate fit, silence ratio, energy CV,
          voiced ratio, spectral flatness, text ease

e2v + DSP    = low-DSP + emotion2vec target probability and entropy
SUPERB + DSP = low-DSP + SUPERB target probability and entropy
VAD + DSP    = low-DSP + VAD target probability and raw VAD
all + DSP    = low-DSP + emotion2vec + SUPERB + VAD target signals
```

## Agreement Results

| surrogate | validation | Spearman | Kendall tau-b | pairwise accuracy | MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| low-DSP | LOOCV | 0.123 | 0.087 | 0.544 | 0.201 |
| low-DSP | leave dataset out | 0.014 | 0.028 | 0.514 | 0.231 |
| low-DSP | leave system out | -0.096 | -0.075 | 0.462 | 0.235 |
| emotion2vec + DSP | LOOCV | 0.378 | 0.228 | 0.614 | 0.155 |
| emotion2vec + DSP | leave dataset out | 0.387 | 0.246 | 0.622 | 0.177 |
| emotion2vec + DSP | leave system out | 0.227 | 0.141 | 0.570 | 0.229 |
| SUPERB + DSP | LOOCV | 0.728 | 0.543 | 0.771 | 0.127 |
| SUPERB + DSP | leave dataset out | 0.585 | 0.418 | 0.709 | 0.169 |
| SUPERB + DSP | leave system out | 0.440 | 0.308 | 0.654 | 0.252 |
| MSP-VAD + DSP | LOOCV | 0.482 | 0.348 | 0.674 | 0.137 |
| MSP-VAD + DSP | leave dataset out | 0.420 | 0.306 | 0.653 | 0.202 |
| MSP-VAD + DSP | leave system out | -0.287 | -0.202 | 0.399 | 0.254 |
| all emotion + DSP | LOOCV | 0.866 | 0.698 | 0.849 | 0.078 |
| all emotion + DSP | leave dataset out | 0.868 | 0.708 | 0.854 | 0.086 |
| all emotion + DSP | leave system out | 0.458 | 0.315 | 0.658 | 0.220 |

The large drop under leave-system-out validation is direct evidence that the
high LOOCV result does not generalize across TTS systems. The all-model
surrogate also reuses signals from its teacher and is metric distillation, not
independent validation. No candidate is currently a drop-in replacement for
the Main metric. `SUPERB + DSP` is the most practical screening candidate.

## Aggregate Sanity Check

| group | n | mean I | mean E | mean Main |
| --- | ---: | ---: | ---: | ---: |
| all clips | 52 | 0.922 | 0.567 | 0.720 |
| non-boundary | 29 | 0.944 | 0.491 | 0.684 |
| boundary | 23 | 0.895 | 0.664 | 0.765 |
| Parler | 40 | 0.915 | 0.518 | 0.679 |
| Bark | 6 | 0.916 | 0.881 | 0.902 |
| SAPI | 6 | 0.976 | 0.581 | 0.807 |

The boundary average being higher than the non-boundary average is not evidence
that boundary audio is better. Many boundary clips request neutral emotion,
which the classifiers score highly, while acoustic sanity fails to resolve
robotic, noisy, and reverberant cases. The Main score currently reflects
automatic intelligibility and requested-emotion agreement to a useful extent,
but it is not a complete TTS naturalness or overall-quality metric. Cross-system
means are additionally confounded by emotion-label composition and should not
be used as system rankings.

## Compute Cost

Warm per-clip cost was measured on the local RTX 4070 SUPER. Emotion and DSP
costs are measured on the 52/44-clip runs. Whisper and sanity use the prior
same-machine 26-clip measurement, so Main cost is a normalized estimate.

| metric | seconds/clip | speedup vs Main |
| --- | ---: | ---: |
| Main auto v3 | 0.7169 | 1.0x |
| low-DSP | 0.0550 | 13.0x |
| emotion2vec + DSP | 0.0922 | 7.8x |
| SUPERB + DSP | 0.0748 | 9.6x |
| MSP-VAD + DSP | 0.0710 | 10.1x |
| all emotion + DSP | 0.1281 | 5.6x |

Model startup and first download are excluded from warm cost. In the first
44-clip run, the MSP-Dim total included a one-time download and was about 132
seconds; a cached eight-clip run loaded it in about 5.3 seconds.

## Reproducible Outputs

- Main implementation: `scripts/analyze_automatic_emotion_consensus.py`
- Emotion inference: `scripts/evaluate_automatic_emotion_models.py`
- Controlled generation input and audio: `experiments/automatic_emotion_consensus_v1/controlled_generation/`
- All 52 per-clip scores: `experiments/automatic_emotion_consensus_v1/analysis/per_clip_scores.csv`
- Agreement table: `experiments/automatic_emotion_consensus_v1/analysis/surrogate_candidates.csv`
- Cost table: `experiments/automatic_emotion_consensus_v1/analysis/metric_costs.csv`
- Generated full report: `experiments/automatic_emotion_consensus_v1/analysis/automatic_metric_report.md`

## Next Improvements

The most useful automatic-only next steps are a stronger second ASR,
verbalization-aware text normalization, a TTS-focused MOS predictor, and an
unseen fourth TTS system with non-neutral emotion control. Human labels remain
necessary before calling the scalar perceptual truth, but they are not required
to reproduce the present automatic-only experiment.

## Appendix: Reference Implementation

The following blocks are the original Python implementation used to produce
the reported results. They are copied from the named source files rather than
rewritten as illustrative pseudocode. The snippets rely on the imports and
audio/text helper functions in those files.

### A. Main Metric Constants and Functions

Source: `scripts/analyze_automatic_emotion_consensus.py`

```python
EMOTIONS = ("neutral", "happy", "angry", "sad")
# Independent, interpretable VAD anchors. They are hypotheses, not fitted labels.
VAD_ANCHORS = {
    "neutral": (0.40, 0.50, 0.50),
    "happy": (0.70, 0.70, 0.75),
    "angry": (0.75, 0.70, 0.25),
    "sad": (0.30, 0.35, 0.30),
}
VAD_SCALES = np.asarray((0.25, 0.25, 0.30), dtype="float64")
RIDGE_ALPHA = 10.0


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def vad_probabilities(row: dict[str, str]) -> dict[str, float]:
    point = np.asarray((f(row, "vad_arousal"), f(row, "vad_dominance"), f(row, "vad_valence")))
    logits = []
    for emotion in EMOTIONS:
        delta = (point - np.asarray(VAD_ANCHORS[emotion])) / VAD_SCALES
        logits.append(-0.5 * float(delta @ delta))
    values = np.exp(np.asarray(logits) - max(logits))
    values /= values.sum()
    return dict(zip(EMOTIONS, values.tolist()))


def median3(a: float, b: float, c: float) -> float:
    return sorted((a, b, c))[1]


def automatic_main(row: dict[str, str]) -> dict[str, float | str]:
    target = row["target_emotion"].lower()
    vad = vad_probabilities(row)
    e2v = f(row, "e2v_target_prob")
    superb = f(row, "superb_target_prob")
    emotion = median3(e2v, superb, vad[target])
    disagreement = max(e2v, superb, vad[target]) - min(e2v, superb, vad[target])
    intelligibility = clamp(1.0 - f(row, "wer"))
    sanity = clamp(f(row, "acoustic_sanity_score_0_1"))
    eps = 1e-6
    score = math.exp(
        0.55 * math.log(max(intelligibility, eps))
        + 0.35 * math.log(max(emotion, eps))
        + 0.10 * math.log(max(sanity, eps))
    )
    eligible = intelligibility >= 0.70 and sanity >= 0.50
    return {
        "intelligibility_auto_0_1": intelligibility,
        "emotion_consensus_0_1": emotion,
        "emotion_model_disagreement_0_1": disagreement,
        "vad_target_prob": vad[target],
        "vad_top_label": max(vad, key=vad.get),
        "acoustic_sanity_0_1": sanity,
        "main_auto_v3_0_1": score,
        "ranking_eligible": "1" if eligible else "0",
    }
```

The input row fields `e2v_target_prob`, `superb_target_prob`, and the three VAD
dimensions are generated by `scripts/evaluate_automatic_emotion_models.py`.

### B. Low-DSP Acoustic Feature Implementation

Source: `surrogate_exploration_v1/analyze_surrogates.py`

```python
def acoustic_features(row: dict[str, str]) -> dict[str, float]:
    import numpy as np

    audio = load_audio_16k(resolve_audio(row["audio_path"]))
    duration = len(audio) / 16_000.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else math.nan
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-8))
    peak_abs = float(np.max(np.abs(audio))) if len(audio) else math.nan
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.98)) if len(audio) else math.nan
    frame_rms = rms_values(audio)
    silence_threshold = max(1e-4, rms * 0.10) if math.isfinite(rms) else 1e-4
    silence_ratio = float(np.mean(frame_rms < silence_threshold)) if frame_rms.size else math.nan
    rms_mean = float(np.mean(frame_rms)) if frame_rms.size else math.nan
    energy_cv = float(np.std(frame_rms) / max(rms_mean, 1e-8)) if math.isfinite(rms_mean) else math.nan
    f0_values = estimate_f0_autocorr(audio)
    f0_std = float(np.std(f0_values)) if f0_values else 0.0
    f0_range = float(np.max(f0_values) - np.min(f0_values)) if f0_values else 0.0
    voiced_ratio = float(len(f0_values) / len(frame_rms)) if frame_rms.size else math.nan
    zcr_values = []
    for frame in frame_audio(audio):
        if len(frame) > 1:
            zcr_values.append(float(np.mean(np.abs(np.diff(np.signbit(frame))))))
    spec = spectral_features(audio)

    loudness_penalty = 0.0
    if rms_dbfs < -32.0:
        loudness_penalty = clamp((-32.0 - rms_dbfs) / 18.0)
    elif rms_dbfs > -8.0:
        loudness_penalty = clamp((rms_dbfs + 8.0) / 8.0)
    silence_penalty = 0.0
    if math.isfinite(silence_ratio):
        if silence_ratio > 0.45:
            silence_penalty = clamp((silence_ratio - 0.45) / 0.35)
        elif silence_ratio < 0.02:
            silence_penalty = clamp((0.02 - silence_ratio) / 0.02)
    clipping_penalty = clamp(clipping_ratio * 100.0) if math.isfinite(clipping_ratio) else 0.0
    flatness_penalty = clamp((spec["spectral_flatness"] - 0.35) / 0.40) if spec["spectral_flatness"] > 0.35 else 0.0
    duration_penalty = clamp((1.5 - duration) / 1.5) if duration < 1.5 else 0.0
    signal_quality = 1.0 - clamp(
        0.30 * loudness_penalty
        + 0.30 * silence_penalty
        + 0.20 * clipping_penalty
        + 0.15 * flatness_penalty
        + 0.05 * duration_penalty
    )

    target_emotion = (row.get("target_emotion") or "neutral").strip().lower()
    word_count = text_difficulty(row.get("text", ""))["word_count"]
    speech_rate_wps = word_count / max(duration, 1e-8)
    rate_target, rate_tolerance = RATE_TARGETS.get(target_emotion, RATE_TARGETS["neutral"])
    rate_fit = clamp(1.0 - abs(speech_rate_wps - rate_target) / rate_tolerance)
    prosody_activity = 0.5 * min(1.0, f0_std / 45.0) + 0.5 * min(1.0, energy_cv / 0.9)
    prosody_target, prosody_tolerance = PROSODY_TARGETS.get(target_emotion, PROSODY_TARGETS["neutral"])
    prosody_fit = clamp(1.0 - abs(prosody_activity - prosody_target) / prosody_tolerance)

    return {
        "duration_sec": duration,
        "speech_rate_wps": speech_rate_wps,
        "rms_dbfs": rms_dbfs,
        "peak_abs": peak_abs,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "energy_cv": energy_cv,
        "f0_std_hz": f0_std,
        "f0_range_hz": f0_range,
        "voiced_ratio": voiced_ratio,
        "zcr": mean(zcr_values),
        "prosody_activity_light": prosody_activity,
        "prosody_fit_light": prosody_fit,
        "rate_fit": rate_fit,
        "signal_quality": signal_quality,
        **spec,
    }
```

The Main/surrogate analysis selects the deployable low-cost subset as follows.
This is the original wrapper from
`scripts/analyze_automatic_emotion_consensus.py`:

```python
def add_low_cost_features(rows: list[dict[str, object]]) -> tuple[float, list[str]]:
    start = time.perf_counter()
    for row in rows:
        cheap = acoustic_features({key: str(value) for key, value in row.items()})
        text = text_difficulty(str(row["text"]))
        for key in ("signal_quality", "rate_fit", "silence_ratio", "energy_cv", "voiced_ratio", "spectral_flatness"):
            row[key] = cheap[key]
        row["text_ease"] = text["text_ease"]
    elapsed = time.perf_counter() - start
    return elapsed, [
        "signal_quality",
        "rate_fit",
        "silence_ratio",
        "energy_cv",
        "voiced_ratio",
        "spectral_flatness",
        "text_ease",
    ]
```

### C. Ridge Fit and Prediction

Source: `surrogate_exploration_v1/analyze_surrogates_v2.py`

```python
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
```

### D. Fold-Pure Validation

Source: `scripts/analyze_automatic_emotion_consensus.py`

```python
def split_predictions(
    rows: list[dict[str, float | str]], target: list[float], features: list[str], split_key: str
) -> list[float]:
    predictions = [math.nan] * len(rows)
    groups = list(range(len(rows))) if split_key == "LOOCV" else sorted({str(row[split_key]) for row in rows})
    for group in groups:
        if split_key == "LOOCV":
            test = [int(group)]
        else:
            test = [i for i, row in enumerate(rows) if str(row[split_key]) == group]
        test_set = set(test)
        train = [i for i in range(len(rows)) if i not in test_set]
        x_train = [[float(rows[i][name]) for name in features] for i in train]
        x_test = [[float(rows[i][name]) for name in features] for i in test]
        coef, center, scale = ridge_fit(x_train, [target[i] for i in train], alpha=RIDGE_ALPHA)
        for idx, prediction in zip(test, ridge_predict(x_test, coef, center, scale)):
            predictions[idx] = prediction
    return predictions
```

This function performs standardization and fitting only on each training fold.
`split_key="dataset"` implements leave-dataset-out and
`split_key="tts_system"` implements leave-system-out.

### E. Surrogate Metric Definitions

The following function is the exact definition of the five fitted surrogate
families and their three validation modes. Source:
`scripts/analyze_automatic_emotion_consensus.py`.

```python
def build_surrogates(rows: list[dict[str, object]], low_features: list[str]) -> tuple[list[dict[str, object]], dict[str, list[float]]]:
    typed = rows  # The numeric feature accesses below are explicit.
    target = [float(row["main_auto_v3_0_1"]) for row in rows]
    feature_sets = {
        "low_dsp_ridge": low_features,
        "e2v_plus_dsp_ridge": low_features + ["e2v_target_prob", "e2v_entropy_norm"],
        "superb_plus_dsp_ridge": low_features + ["superb_target_prob", "superb_entropy_norm"],
        "vad_plus_dsp_ridge": low_features + ["vad_target_prob", "vad_arousal", "vad_dominance", "vad_valence"],
        "all_emotion_plus_dsp_ridge": low_features
        + ["e2v_target_prob", "e2v_entropy_norm", "superb_target_prob", "superb_entropy_norm", "vad_target_prob"],
    }
    results: list[dict[str, object]] = []
    predictions: dict[str, list[float]] = {}
    for name, features in feature_sets.items():
        for validation, split_key in (("LOOCV", "LOOCV"), ("leave_dataset_out", "dataset"), ("leave_system_out", "tts_system")):
            if split_key != "LOOCV" and len({str(row[split_key]) for row in rows}) < 2:
                continue
            pred = split_predictions(typed, target, features, split_key)  # type: ignore[arg-type]
            key = f"{name}__{validation}"
            predictions[key] = pred
            row = evaluate(name, pred, target, "", "+".join(features), f"alpha={RIDGE_ALPHA:g}")
            row["validation"] = validation
            results.append(row)
    return results, predictions
```

The fitted coefficients are fold-specific and are intentionally not presented
as one deployable coefficient vector. A production surrogate requires a frozen
training set, one final fit, and validation on an additional unseen TTS system.
