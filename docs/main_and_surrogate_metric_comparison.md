# Main Metric and SIM-like Surrogate Metric Comparison

Current version: 2026-07-19

This document describes the current automatic main metric and the best tested
SIM-like surrogate metric for the emotion-aware TTS benchmark. All scores are
normalized to `[0, 1]`, where higher is better.

## Data Used

The comparison uses the current 26 local generated audio samples:

```text
experiments/parler_emotion_v1/combined/parler_emotion_scored_main_metric.csv
experiments/boundary_metric_v1/combined/boundary_scored_main_metric.csv
```

The surrogate results are stored in:

```text
surrogate_exploration_v1/outputs_v3/sim_like/
```

## Main Metric

The main metric is the reference target. It combines text preservation,
acoustic usability, emotion match, and prosody fit.

### Inputs

| symbol | source field | meaning |
| --- | --- | --- |
| `WER` | `wer` | ASR word error rate |
| `CER` | `cer` | ASR character error rate |
| `naturalness_proxy_1_5` | `naturalness_proxy_1_5` | no-reference acoustic quality proxy |
| `target_emotion_prob` | `target_emotion_prob` | SER probability assigned to the target emotion |
| `target_emotion_match` | `target_emotion_match` | `1` if SER top label equals target emotion, else `0` |
| `prosody_activity` | `prosody_activity_0_1` | pitch/energy activity score |

### Formula

```text
clip(x) = min(max(x, 0), 1)

I = clip(0.80 * (1 - WER) + 0.20 * (1 - CER))
Q = clip((naturalness_proxy_1_5 - 1) / 4)
E = clip(0.70 * target_emotion_prob + 0.30 * target_emotion_match)
P = clip(1 - abs(prosody_activity - target_prosody) / tolerance)

raw = clip(0.45 * I + 0.15 * Q + 0.30 * E + 0.10 * P)
gate = clip(0.35 + 0.65 * I)

main_metric = clip(raw * gate)
```

Prosody targets:

| emotion | target prosody activity | tolerance |
| --- | ---: | ---: |
| happy | 0.85 | 0.35 |
| angry | 0.90 | 0.35 |
| sad | 0.60 | 0.40 |
| neutral | 0.80 | 0.40 |

### Interpretation

The main metric is intentionally more expensive because it uses ASR and SER:

- ASR/Whisper contributes WER and CER.
- SER contributes target emotion probability and top-label match.
- Lightweight DSP contributes acoustic naturalness and prosody activity.

This score is not treated as perfect human MOS. It is the current automatic
benchmark target for surrogate search.

## SIM-like Surrogate Metric

The best tested surrogate is:

```text
ridge_sim_ser_plus_low_dsp_loo
```

It combines:

- neural target-emotion probability/logits;
- SIM-like wav2vec2 embedding margin;
- same-emotion embedding consistency;
- low-cost DSP and text features.

Model used for the neural signal:

```text
superb/wav2vec2-base-superb-er
```

Important caveat: this is not a production speaker-SIM metric with a separate
reference speaker recording. It is a local SIM-like neural audio embedding
proxy. Raw centroid cosine alone did not fit the main metric well. The useful
surrogate combines neural target-emotion probability with embedding margin and
low-DSP features.

### Neural Embedding Features

Let:

```text
e = L2-normalized wav2vec2 hidden embedding for the current audio
c_target = centroid embedding for samples with the same target emotion
c_other = centroid embedding for a non-target emotion
cos(a, b) = dot(a, b) / (||a|| * ||b||)
```

For held-out evaluation, centroids are computed without the held-out sample.

```text
sim_target_cos = clip((cos(e, c_target) + 1) / 2)
```

```text
sim_target_margin = clip(
  0.5 + (cos(e, c_target) - max_other cos(e, c_other)) / 0.10
)
```

```text
sim_pairwise_consistency = clip(
  0.5 + (mean_same_emotion_cosine - mean_different_emotion_cosine) / 0.10
)
```

The target-emotion probability is taken from the same wav2vec2 classifier:

```text
sim_ser_target_prob = softmax(logits)[target_emotion]
```

### Low-DSP/Text Features

The surrogate also uses these low-cost features:

```text
fit(x, target, tolerance) = clip(1 - abs(x - target) / tolerance)
norm_range(x, low, high) = clip((x - low) / (high - low))
```

```text
prosody_activity_light =
  0.5 * min(1, f0_std_hz / 45)
+ 0.5 * min(1, energy_cv / 0.9)
```

```text
prosody_fit_light = fit(
  prosody_activity_light,
  emotion_prosody_target,
  emotion_prosody_tolerance
)
```

```text
speech_rate_wps = word_count / duration_sec
rate_fit = fit(speech_rate_wps, emotion_rate_target, emotion_rate_tolerance)
```

```text
text_difficulty = clip(
  0.10 * acronym_count
+ 0.035 * number_word_count
+ 0.45 * repeated_ratio
+ 1.40 * max(0, sibilant_density - 0.12)
+ 0.25 * max(0, max_initial_ratio - 0.35)
+ 1.00 * max(0, punctuation_ratio - 0.06)
)

text_ease = 1 - text_difficulty
```

The target-style feature compares the audio against a simple target-emotion
profile:

| emotion | activity | rate | loudness | pitch range | energy CV | pause rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| happy | 0.82 | 3.25 | -17.0 | 290 | 0.88 | 0.20 |
| angry | 0.88 | 3.65 | -17.5 | 300 | 0.95 | 0.18 |
| sad | 0.28 | 2.65 | -20.5 | 185 | 0.62 | 0.45 |
| neutral | 0.55 | 3.15 | -18.5 | 235 | 0.78 | 0.28 |

```text
activity_fit = fit(prosody_activity_light, target.activity, 0.42)
rate_fit_emotion = fit(speech_rate_wps, target.rate, 1.25)
loudness_fit = fit(rms_dbfs, target.loudness, 7.0)
pitch_fit = fit(f0_range_hz, target.pitch_range, 175.0)
energy_fit = fit(energy_cv, target.energy_cv, 0.55)
pause_fit = fit(pause_rate_per_sec, target.pause_rate, 0.65)

target_style_fit_v1 = clip(
  0.28 * activity_fit
+ 0.18 * rate_fit_emotion
+ 0.16 * pitch_fit
+ 0.16 * energy_fit
+ 0.12 * loudness_fit
+ 0.10 * pause_fit
)
```

### Ridge Surrogate Formula

The best surrogate is a ridge regression over these features:

```text
features =
  sim_ser_target_prob
  sim_target_margin
  sim_pairwise_consistency
  target_style_fit_v1
  prosody_fit_light
  rate_fit
  text_ease
```

```text
z_i = (feature_i - mean_i) / std_i

surrogate_metric = clip(
  intercept + sum(coef_i * z_i)
)
```

The reported agreement scores use leave-one-out cross validation. The following
coefficients are a reference all-data fit for the current 26-sample set:

| feature | coefficient | mean | std |
| --- | ---: | ---: | ---: |
| intercept | 0.781784 |  |  |
| `sim_ser_target_prob` | 0.142486 | 0.550498 | 0.379609 |
| `sim_target_margin` | -0.029542 | 0.477146 | 0.035178 |
| `sim_pairwise_consistency` | 0.030845 | 0.482990 | 0.146414 |
| `target_style_fit_v1` | 0.009679 | 0.584651 | 0.159092 |
| `prosody_fit_light` | 0.012134 | 0.563191 | 0.259785 |
| `rate_fit` | -0.007257 | 0.379064 | 0.309589 |
| `text_ease` | 0.079559 | 0.949742 | 0.115322 |

## Agreement Results

Agreement is measured against `main_metric_0_1`.

| surrogate | Pearson | Spearman | MAE | top5 overlap | bottom5 overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ridge_sim_ser_plus_low_dsp_loo` | 0.943748 | 0.928205 | 0.039488 | 0.800000 | 0.800000 |
| `lodo_sim_ser_plus_low_dsp` | 0.720266 | 0.739272 | 0.072021 | 0.600000 | 0.600000 |
| `sim_ser_target_prob` | 0.745731 | 0.737436 | 0.275427 | 0.800000 | 0.600000 |
| `ridge_sim_plus_low_dsp_loo` | 0.473116 | 0.402393 | 0.108157 | 0.000000 | 0.600000 |
| `ridge_sim_centroid_loo` | 0.002307 | 0.029060 | 0.148841 | 0.200000 | 0.200000 |

Interpretation:

- The best SIM-like surrogate matches the current main metric closely on this
  small local set.
- Raw embedding centroid cosine is not sufficient.
- Neural target-emotion probability is the dominant useful neural signal.
- Leave-one-dataset-out performance is lower than LOOCV, so more data is needed
  before freezing the surrogate as a final benchmark metric.

## Per-Sample Score Comparison

| audio id | target emotion | main metric | surrogate metric | absolute error |
| --- | --- | ---: | ---: | ---: |
| `neutral_01` | neutral | 0.962698 | 0.949480 | 0.013218 |
| `happy_02` | happy | 0.952333 | 0.962066 | 0.009733 |
| `happy_01` | happy | 0.916491 | 0.944932 | 0.028441 |
| `neutral_02` | neutral | 0.891419 | 0.945089 | 0.053670 |
| `angry_01` | angry | 0.851983 | 0.794196 | 0.057787 |
| `angry_02` | angry | 0.642522 | 0.714542 | 0.072020 |
| `sad_02` | sad | 0.622155 | 0.613814 | 0.008341 |
| `sad_01` | sad | 0.560777 | 0.618625 | 0.057848 |
| `function_word_repetition` | neutral | 0.993317 | 1.000000 | 0.006683 |
| `sad_text_happy_voice` | happy | 0.966175 | 0.999787 | 0.033612 |
| `digits_address` | neutral | 0.949625 | 0.799288 | 0.150337 |
| `angry_text_neutral_voice` | neutral | 0.908341 | 0.877899 | 0.030442 |
| `homophones_minimal_pairs` | neutral | 0.879373 | 0.868668 | 0.010705 |
| `robotic_monotone` | neutral | 0.854820 | 0.882234 | 0.027414 |
| `noisy_neutral` | neutral | 0.851358 | 0.908678 | 0.057320 |
| `exaggerated_happy` | happy | 0.850926 | 0.845565 | 0.005361 |
| `control_neutral_neutral` | neutral | 0.849007 | 0.809095 | 0.039912 |
| `distant_reverb_neutral` | neutral | 0.843101 | 0.874110 | 0.031009 |
| `control_angry_angry` | angry | 0.835137 | 0.746928 | 0.088209 |
| `control_happy_happy` | happy | 0.678977 | 0.721561 | 0.042584 |
| `whisper_sad` | sad | 0.616864 | 0.599924 | 0.016940 |
| `neutral_text_angry_voice` | angry | 0.610671 | 0.580583 | 0.030088 |
| `happy_text_sad_voice` | sad | 0.604236 | 0.586127 | 0.018109 |
| `control_sad_sad` | sad | 0.600534 | 0.583997 | 0.016537 |
| `technical_acronyms` | neutral | 0.551019 | 0.576949 | 0.025930 |
| `fast_tongue_twister` | neutral | 0.482527 | 0.576969 | 0.094442 |

Largest errors:

| audio id | absolute error | likely reason |
| --- | ---: | --- |
| `digits_address` | 0.150337 | ASR-sensitive number/address content is not fully captured by emotion/style signals |
| `fast_tongue_twister` | 0.094442 | fast articulation and intelligibility stress still need ASR-like evidence |
| `control_angry_angry` | 0.088209 | angry style is less stable for the neural surrogate |
| `angry_02` | 0.072020 | main metric penalizes emotion mismatch more strongly |

## Efficiency Comparison

Measured locally on the same 26 clips with cached models and CUDA.

| metric/scenario | total seconds / 26 clips | seconds / clip | relative to main | approximate speedup |
| --- | ---: | ---: | ---: | ---: |
| main metric current pipeline | 56.804452 | 2.184787 | 1.000000 | 1.0x |
| low-DSP surrogate features | 2.362031 | 0.090847 | 0.041582 | 24.0x |
| SIM-like embedding only | 3.904094 | 0.150157 | 0.068729 | 14.5x |
| SIM-like + low-DSP surrogate | 4.600160 | 0.176929 | 0.080982 | 12.3x |

Main metric component costs:

| component | total seconds / 26 clips | seconds / clip | share of main |
| --- | ---: | ---: | ---: |
| SER emotion/prosody | 32.084681 | 1.234026 | 56.5% |
| Whisper WER/CER | 23.426323 | 0.901012 | 41.2% |
| naturalness proxy | 1.186913 | 0.045651 | 2.1% |
| final composite scoring | 0.106535 | 0.004098 | 0.2% |

## Practical Conclusion

The current SIM-like surrogate is the best tested approximation to the main
metric:

```text
Pearson  = 0.943748
Spearman = 0.928205
MAE      = 0.039488
Cost     = 8.10% of the current main metric pipeline
Speedup  = about 12.3x
```

It is a useful candidate for fast autoresearch or screening runs, especially
when full Whisper WER evaluation is too expensive. However, it should not yet be
treated as final. The current weaknesses are ASR-sensitive cases such as
numbers, acronyms, and fast tongue-twisters, plus lower leave-dataset-out
stability on the small 26-sample set.

## Project Implementation Notes

This section maps the metric design above to the current source code.

### Main Metric Implementation

Main script:

```text
scripts/score_emotion_tts_main_metric.py
```

Core functions and variables:

| code item | role |
| --- | --- |
| `PROSODY_TARGETS` | target prosody activity and tolerance for `happy`, `angry`, `sad`, and `neutral` |
| `prosody_fit(target_emotion, prosody_activity)` | computes the clipped target-dependent prosody score `P` |
| `score_row(row)` | computes `I`, `Q`, `E`, `P`, `raw`, `gate`, and `main_metric` for one row |
| `main()` | reads the combined metric CSV, applies `score_row`, writes CSV and Markdown outputs |

Inside `score_row(row)`, the source fields are read as:

```python
wer = clamp(parse_float(row, "wer"), 0.0, 1.0)
cer = clamp(parse_float(row, "cer"), 0.0, 1.0)
naturalness_1_5 = parse_float(row, "naturalness_proxy_1_5")
target_prob = clamp(parse_float(row, "target_emotion_prob"), 0.0, 1.0)
target_match = clamp(parse_float(row, "target_emotion_match"), 0.0, 1.0)
prosody_activity = clamp(parse_float(row, "prosody_activity_0_1"), 0.0, 1.0)
```

The component variables are:

```python
intelligibility = clamp(0.80 * (1.0 - wer) + 0.20 * (1.0 - cer))
naturalness = clamp((naturalness_1_5 - 1.0) / 4.0)
emotion_match = clamp(0.70 * target_prob + 0.30 * target_match)
prosody = prosody_fit(target_emotion, prosody_activity)
```

The final score is implemented as:

```python
weighted_raw = clamp(
    0.45 * intelligibility
    + 0.15 * naturalness
    + 0.30 * emotion_match
    + 0.10 * prosody
)
intelligibility_gate = clamp(0.35 + 0.65 * intelligibility)
main_metric = clamp(weighted_raw * intelligibility_gate)
```

The output fields written by `score_row(row)` include:

```text
main_metric_0_1
main_metric_raw_0_1
intelligibility_component_0_1
naturalness_component_0_1
emotion_component_0_1
prosody_fit_component_0_1
intelligibility_gate_0_1
main_metric_formula
```

### Main Metric Upstream Components

The main score consumes outputs from these scripts:

| script | output used by main metric |
| --- | --- |
| `scripts/evaluate_wer_with_transformers_whisper.py` | `wer`, `cer`, `asr_transcript` |
| `scripts/evaluate_acoustic_naturalness_proxy.py` | `naturalness_proxy_1_5` |
| `scripts/evaluate_emotion_prosody.py` | `target_emotion_prob`, `target_emotion_match`, `prosody_activity_0_1` |
| `scripts/build_main_metrics_report.py` | merges intermediate metric outputs into combined reports |

Current models used in the timing experiment:

```text
ASR: openai/whisper-tiny.en
SER: superb/wav2vec2-base-superb-er
```

### Surrogate Metric Implementation

SIM-like surrogate script:

```text
surrogate_exploration_v1/analyze_sim_like_surrogates.py
```

Related low-DSP feature scripts:

```text
surrogate_exploration_v1/analyze_surrogates.py
surrogate_exploration_v1/analyze_surrogates_v3.py
```

Key variables:

| variable | value / meaning |
| --- | --- |
| `MODEL_NAME` | `superb/wav2vec2-base-superb-er` |
| `EMOTIONS` | `["neutral", "happy", "angry", "sad"]` |
| `OUTPUT_DIR` | `surrogate_exploration_v1/outputs_v3/sim_like` |
| `prediction_map` | dictionary of per-sample surrogate predictions |
| `ridge_specs` | feature sets used for LOOCV and leave-dataset-out ridge evaluation |

The neural model is loaded in `setup_model(device)`:

```python
feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)
```

The embeddings and target-emotion probabilities are computed in
`extract_embeddings(rows)`:

```python
outputs = model(**inputs, output_hidden_states=True)
hidden = outputs.hidden_states[-1][0]
pooled = hidden.mean(dim=0)  # or attention-mask weighted mean
pooled_np = pooled / ||pooled||
target_prob = softmax(outputs.logits)[target_emotion]
```

The SIM-like centroid features are computed in `loo_centroid_features(rows,
embeddings)`:

```text
sim_target_cos_loo
sim_target_margin_loo
sim_target_rank_loo
sim_target_softmax_loo
```

The pairwise embedding feature is computed by `pairwise_consistency(rows,
embeddings)` and stored as:

```text
sim_pairwise_consistency
```

The neural classifier probability is attached in `build_rows(...)` as:

```text
sim_ser_target_prob
```

### Surrogate Feature Set in Code

The best current surrogate candidate is defined in `ridge_specs`:

```python
"ridge_sim_ser_plus_low_dsp_loo": [
    "sim_ser_target_prob",
    "sim_target_margin_loo",
    "sim_pairwise_consistency",
    "target_style_fit_v1",
    "prosody_fit_light",
    "rate_fit",
    "text_ease",
]
```

The stricter dataset-transfer check uses the same feature list:

```python
"lodo_sim_ser_plus_low_dsp": [
    "sim_ser_target_prob",
    "sim_target_margin_loo",
    "sim_pairwise_consistency",
    "target_style_fit_v1",
    "prosody_fit_light",
    "rate_fit",
    "text_ease",
]
```

Evaluation dispatch:

```python
if name.startswith("lodo_"):
    preds = ridge_leave_dataset_out(rows, target, features, alpha=5e-2)
else:
    preds = ridge_loo(rows, target, features, alpha=5e-2)
```

The target is:

```python
target = [row["main_metric_0_1"] for row in rows]
```

### Low-DSP Feature Construction in Code

Base low-DSP features are produced by:

```text
analyze_surrogates.text_difficulty(row["text"])
analyze_surrogates.acoustic_features(row)
```

Important generated variables:

```text
text_ease
rate_fit
prosody_activity_light
prosody_fit_light
signal_quality
energy_cv
f0_std_hz
f0_range_hz
voiced_ratio
spectral_centroid_hz
high_freq_ratio
```

Additional v3 waveform/style features are produced by:

```text
analyze_surrogates_v3.audio_shape_features(audio, sample_rate)
analyze_surrogates_v3.add_derived_features(row)
analyze_surrogates_v3.load_rows_with_audio_features()
```

Important generated variables:

```text
target_style_fit_v1
emotion_arousal_fit_v1
pause_naturalness
envelope_stability
dynamic_range_fit
spectral_balance_fit
voice_presence_fit
articulation_risk_inverse
delivery_fit_v1
```

### Surrogate Outputs

The SIM-like script writes:

| output file | content |
| --- | --- |
| `sim_like_candidates.csv` | aggregate agreement metrics for each candidate |
| `sim_like_per_sample.csv` | per-sample main score, surrogate score, and absolute error |
| `sim_like_costs.csv` | measured runtime versus the main metric pipeline |
| `sim_like_report.md` | concise report generated from the same results |

The main fields used from `sim_like_per_sample.csv` for this document are:

```text
main_metric_0_1
ridge_sim_ser_plus_low_dsp_loo
ridge_sim_ser_plus_low_dsp_loo_abs_error
```

### Reproduction Commands

Run the SIM-like surrogate experiment:

```powershell
conda run -n mlevolve-win python surrogate_exploration_v1\analyze_sim_like_surrogates.py
```

Run the metric cost measurement:

```powershell
conda run -n mlevolve-win python surrogate_exploration_v1\measure_metric_costs.py
```

Run the main metric scorer on an already combined metrics CSV:

```powershell
conda run -n mlevolve-win python scripts\score_emotion_tts_main_metric.py `
  --input path\to\combined_metrics.csv `
  --output-csv path\to\scored_main_metric.csv `
  --output-md path\to\scored_main_metric.md `
  --experiment-name emotion_tts
```

## Raw Source Implementation Appendix

The following snippets are copied from the current local implementation.

### Main Metric Source

Source file:

```text
scripts/score_emotion_tts_main_metric.py
```

```python
PROSODY_TARGETS = {
    "happy": (0.85, 0.35),
    "angry": (0.90, 0.35),
    "sad": (0.60, 0.40),
    "neutral": (0.80, 0.40),
}


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
```

### SIM-like Surrogate Source

Source file:

```text
surrogate_exploration_v1/analyze_sim_like_surrogates.py
```

```python
OUTPUT_DIR = EXPLORE_DIR / "outputs_v3" / "sim_like"
MODEL_NAME = "superb/wav2vec2-base-superb-er"
EMOTIONS = ["neutral", "happy", "angry", "sad"]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def read_audio_16k(path: Path) -> np.ndarray:
    sample_rate, data = wavfile.read(path)
    audio = np.asarray(data)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype("float32") / float(np.iinfo(data.dtype).max)
    else:
        audio = np.clip(audio.astype("float32"), -1.0, 1.0)
    if sample_rate != 16_000:
        gcd = math.gcd(sample_rate, 16_000)
        audio = resample_poly(audio, 16_000 // gcd, sample_rate // gcd).astype("float32")
    return audio


def setup_model(device: str) -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME, local_files_only=True)
    model.to(device)
    model.eval()
    return {
        "torch": torch,
        "feature_extractor": feature_extractor,
        "model": model,
        "device": device,
    }


def target_prob_from_logits(row: dict[str, float], logits: np.ndarray, id2label: dict[int, str]) -> float:
    exp = np.exp(logits - np.max(logits))
    probs = exp / max(float(exp.sum()), 1e-12)
    target = str(row.get("target_emotion", "neutral")).lower()
    aliases = {
        "neutral": {"neu", "neutral"},
        "happy": {"hap", "happy"},
        "angry": {"ang", "angry"},
        "sad": {"sad"},
    }
    best = 0.0
    for idx, label in id2label.items():
        normalized = str(label).lower()
        if normalized in aliases.get(target, {target}):
            best = max(best, float(probs[int(idx)]))
    return best


def extract_embeddings(rows: list[dict[str, float]]) -> tuple[np.ndarray, list[float], float, str]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_paths = load_audio_paths()

    start = time.perf_counter()
    modules = setup_model(device)
    model = modules["model"]
    feature_extractor = modules["feature_extractor"]
    id2label = dict(model.config.id2label)

    embeddings = []
    target_probs = []
    with modules["torch"].no_grad():
        for row in rows:
            audio = read_audio_16k(resolve_audio_path(audio_paths[(row["dataset"], row["id"])]))
            inputs = feature_extractor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0]
            mask = inputs.get("attention_mask")
            if mask is not None:
                frame_mask = mask[0].to(hidden.device).float()
                if frame_mask.shape[0] != hidden.shape[0]:
                    frame_mask = torch.nn.functional.interpolate(
                        frame_mask[None, None, :], size=hidden.shape[0], mode="nearest"
                    )[0, 0]
                pooled = (hidden * frame_mask[:, None]).sum(dim=0) / torch.clamp(frame_mask.sum(), min=1.0)
            else:
                pooled = hidden.mean(dim=0)
            pooled_np = pooled.detach().float().cpu().numpy()
            norm = float(np.linalg.norm(pooled_np))
            if norm > 1e-12:
                pooled_np = pooled_np / norm
            embeddings.append(pooled_np)
            target_probs.append(target_prob_from_logits(row, outputs.logits[0].detach().cpu().numpy(), id2label))

    elapsed = time.perf_counter() - start
    return np.vstack(embeddings), target_probs, elapsed, device


def centroid(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.zeros(1, dtype="float32")
    center = np.mean(np.vstack(vectors), axis=0)
    norm = float(np.linalg.norm(center))
    if norm > 1e-12:
        center = center / norm
    return center


def loo_centroid_features(rows: list[dict[str, float]], embeddings: np.ndarray) -> dict[str, list[float]]:
    target_cos = []
    target_margin = []
    target_rank_score = []
    target_softmax_score = []

    for held_out, row in enumerate(rows):
        train_idx = [idx for idx in range(len(rows)) if idx != held_out]
        centroids = {}
        for emotion in EMOTIONS:
            vectors = [
                embeddings[idx]
                for idx in train_idx
                if str(rows[idx].get("target_emotion", "")).lower() == emotion
            ]
            centroids[emotion] = centroid(vectors)

        sims = {emotion: cosine(embeddings[held_out], center) for emotion, center in centroids.items() if center.size == embeddings.shape[1]}
        target = str(row.get("target_emotion", "neutral")).lower()
        target_value = sims.get(target, 0.0)
        other_values = [value for emotion, value in sims.items() if emotion != target]
        other_max = max(other_values) if other_values else 0.0
        sorted_sims = sorted(sims.items(), key=lambda item: item[1], reverse=True)
        rank = next((idx for idx, (emotion, _value) in enumerate(sorted_sims) if emotion == target), len(sorted_sims) - 1)
        exp_values = np.exp(np.asarray(list(sims.values()), dtype="float64") * 20.0)
        softmax_den = float(exp_values.sum())
        target_idx = list(sims.keys()).index(target) if target in sims else 0
        target_soft = float(exp_values[target_idx] / softmax_den) if softmax_den > 0 else 0.0

        target_cos.append(clamp((target_value + 1.0) / 2.0))
        target_margin.append(clamp(0.5 + (target_value - other_max) / 0.10))
        target_rank_score.append(clamp(1.0 - rank / max(len(sorted_sims) - 1, 1)))
        target_softmax_score.append(clamp(target_soft))

    return {
        "sim_target_cos_loo": target_cos,
        "sim_target_margin_loo": target_margin,
        "sim_target_rank_loo": target_rank_score,
        "sim_target_softmax_loo": target_softmax_score,
    }


def pairwise_consistency(rows: list[dict[str, float]], embeddings: np.ndarray) -> list[float]:
    scores = []
    for idx, row in enumerate(rows):
        target = str(row.get("target_emotion", "neutral")).lower()
        same = []
        diff = []
        for jdx, other in enumerate(rows):
            if idx == jdx:
                continue
            value = cosine(embeddings[idx], embeddings[jdx])
            if str(other.get("target_emotion", "")).lower() == target:
                same.append(value)
            else:
                diff.append(value)
        if not same or not diff:
            scores.append(0.5)
        else:
            scores.append(clamp(0.5 + (mean(same) - mean(diff)) / 0.10))
    return scores


def build_rows(rows: list[dict[str, float]], sim_features: dict[str, list[float]], target_probs: list[float]) -> list[dict[str, float]]:
    out = []
    for idx, row in enumerate(rows):
        item = dict(row)
        for key, values in sim_features.items():
            item[key] = values[idx]
        item["sim_pairwise_consistency"] = sim_features["sim_pairwise_consistency"][idx]
        item["sim_ser_target_prob"] = target_probs[idx]
        out.append(item)
    return out
```

```python
ridge_specs = {
    "ridge_sim_centroid_loo": [
        "sim_target_cos_loo",
        "sim_target_margin_loo",
        "sim_target_rank_loo",
        "sim_target_softmax_loo",
        "sim_pairwise_consistency",
    ],
    "ridge_sim_plus_low_dsp_loo": [
        "sim_target_cos_loo",
        "sim_target_margin_loo",
        "sim_pairwise_consistency",
        "target_style_fit_v1",
        "prosody_fit_light",
        "rate_fit",
        "text_ease",
    ],
    "ridge_sim_ser_plus_low_dsp_loo": [
        "sim_ser_target_prob",
        "sim_target_margin_loo",
        "sim_pairwise_consistency",
        "target_style_fit_v1",
        "prosody_fit_light",
        "rate_fit",
        "text_ease",
    ],
    "lodo_sim_plus_low_dsp": [
        "sim_target_cos_loo",
        "sim_target_margin_loo",
        "sim_pairwise_consistency",
        "target_style_fit_v1",
        "prosody_fit_light",
        "rate_fit",
        "text_ease",
    ],
    "lodo_sim_ser_plus_low_dsp": [
        "sim_ser_target_prob",
        "sim_target_margin_loo",
        "sim_pairwise_consistency",
        "target_style_fit_v1",
        "prosody_fit_light",
        "rate_fit",
        "text_ease",
    ],
}

for name, features in ridge_specs.items():
    if name.startswith("lodo_"):
        preds = ridge_leave_dataset_out(rows, target, features, alpha=5e-2)
        notes = "leave-one-dataset-out"
    else:
        preds = ridge_loo(rows, target, features, alpha=5e-2)
        notes = "LOOCV ridge"
    prediction_map[name] = preds
    candidates.append(evaluate(name, preds, target, "medium_neural", "+".join(features), notes))
```

### Low-DSP Feature Source

Source file:

```text
surrogate_exploration_v1/analyze_surrogates_v3.py
```

```python
def fit_to_target(value: float, target: float, tolerance: float) -> float:
    return clamp(1.0 - abs(value - target) / max(tolerance, 1e-8))


def norm_range(value: float, low: float, high: float) -> float:
    return clamp((value - low) / max(high - low, 1e-8))


def audio_shape_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    if audio.size == 0:
        return {
            "pause_count": 0.0,
            "pause_rate_per_sec": 0.0,
            "active_ratio_env": 0.0,
            "dynamic_range_db": 0.0,
            "envelope_jitter": 0.0,
            "spectral_rolloff_hz": 0.0,
            "spectral_bandwidth_hz": 0.0,
            "low_band_ratio": 0.0,
            "mid_band_ratio": 0.0,
            "very_high_band_ratio": 0.0,
        }

    frames = framed(audio, sample_rate)
    window = np.hanning(frames.shape[1])
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-8)
    active_threshold = max(-55.0, float(np.percentile(db, 80)) - 28.0)
    active = db > active_threshold
    hop_sec = 0.010
    pause_count, pause_total = silence_segments(active, hop_sec)
    duration = audio.size / float(sample_rate)

    spectra = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)
    power = spectra + 1e-10
    power_sum = power.sum(axis=1)
    centroid = (power * freqs).sum(axis=1) / power_sum
    bandwidth = np.sqrt(((freqs - centroid[:, None]) ** 2 * power).sum(axis=1) / power_sum)
    cdf = np.cumsum(power, axis=1) / power_sum[:, None]
    rolloff_idx = np.argmax(cdf >= 0.85, axis=1)
    rolloff = freqs[rolloff_idx]

    total_power = float(power.sum())
    low_band = float(power[:, freqs < 300.0].sum()) / total_power
    mid_band = float(power[:, (freqs >= 300.0) & (freqs < 3000.0)].sum()) / total_power
    very_high_band = float(power[:, freqs >= 6000.0].sum()) / total_power

    return {
        "pause_count": float(pause_count),
        "pause_rate_per_sec": pause_count / max(duration, 1e-8),
        "active_ratio_env": float(np.mean(active)),
        "dynamic_range_db": float(np.percentile(db, 95) - np.percentile(db, 10)),
        "envelope_jitter": float(np.mean(np.abs(np.diff(db))) / 20.0) if len(db) > 1 else 0.0,
        "spectral_rolloff_hz": float(np.mean(rolloff)),
        "spectral_bandwidth_hz": float(np.mean(bandwidth)),
        "low_band_ratio": low_band,
        "mid_band_ratio": mid_band,
        "very_high_band_ratio": very_high_band,
        "long_pause_total_sec": pause_total,
    }


def style_targets(target_emotion: str) -> dict[str, float]:
    profiles = {
        "happy": {
            "activity": 0.82,
            "rate": 3.25,
            "loudness": -17.0,
            "pitch_range": 290.0,
            "energy_cv": 0.88,
            "pause_rate": 0.20,
            "spectral_centroid": 740.0,
        },
        "angry": {
            "activity": 0.88,
            "rate": 3.65,
            "loudness": -17.5,
            "pitch_range": 300.0,
            "energy_cv": 0.95,
            "pause_rate": 0.18,
            "spectral_centroid": 760.0,
        },
        "sad": {
            "activity": 0.28,
            "rate": 2.65,
            "loudness": -20.5,
            "pitch_range": 185.0,
            "energy_cv": 0.62,
            "pause_rate": 0.45,
            "spectral_centroid": 520.0,
        },
        "neutral": {
            "activity": 0.55,
            "rate": 3.15,
            "loudness": -18.5,
            "pitch_range": 235.0,
            "energy_cv": 0.78,
            "pause_rate": 0.28,
            "spectral_centroid": 620.0,
        },
    }
    return profiles.get(target_emotion.lower(), profiles["neutral"])


def add_derived_features(row: dict[str, float]) -> None:
    target = style_targets(str(row.get("target_emotion", "neutral")))
    rate = row["speech_rate_wps"]
    activity = row["prosody_activity_light"]
    pitch_range = row["f0_range_hz"]
    energy_cv = row["energy_cv"]
    loudness = row["rms_dbfs"]
    pause_rate = row["pause_rate_per_sec"]
    centroid = row["spectral_centroid_hz"]

    activity_fit = fit_to_target(activity, target["activity"], 0.42)
    rate_fit_emotion = fit_to_target(rate, target["rate"], 1.25)
    loudness_fit = fit_to_target(loudness, target["loudness"], 7.0)
    pitch_fit = fit_to_target(pitch_range, target["pitch_range"], 175.0)
    energy_fit = fit_to_target(energy_cv, target["energy_cv"], 0.55)
    pause_fit = fit_to_target(pause_rate, target["pause_rate"], 0.65)
    centroid_fit = fit_to_target(centroid, target["spectral_centroid"], 650.0)

    row["emotion_arousal_fit_v1"] = clamp(
        0.35 * activity_fit
        + 0.20 * energy_fit
        + 0.15 * pitch_fit
        + 0.15 * loudness_fit
        + 0.15 * rate_fit_emotion
    )
    row["target_style_fit_v1"] = clamp(
        0.28 * activity_fit
        + 0.18 * rate_fit_emotion
        + 0.16 * pitch_fit
        + 0.16 * energy_fit
        + 0.12 * loudness_fit
        + 0.10 * pause_fit
    )
    row["pause_naturalness"] = clamp(
        0.50 * fit_to_target(row["silence_ratio"], 0.16, 0.18)
        + 0.35 * pause_fit
        + 0.15 * fit_to_target(row["active_ratio_env"], 0.72, 0.25)
    )
    row["envelope_stability"] = clamp(
        0.55 * fit_to_target(row["envelope_jitter"], 0.08, 0.10)
        + 0.45 * fit_to_target(row["dynamic_range_db"], 22.0, 18.0)
    )
    row["dynamic_range_fit"] = fit_to_target(row["dynamic_range_db"], 22.0, 20.0)
    row["spectral_balance_fit"] = clamp(
        0.45 * centroid_fit
        + 0.25 * fit_to_target(row["high_freq_ratio"], 0.06, 0.08)
        + 0.20 * fit_to_target(row["very_high_band_ratio"], 0.018, 0.035)
        + 0.10 * fit_to_target(row["low_band_ratio"], 0.22, 0.20)
    )
    row["voice_presence_fit"] = clamp(
        0.45 * fit_to_target(row["mid_band_ratio"], 0.62, 0.28)
        + 0.35 * fit_to_target(row["voiced_ratio"], 0.64, 0.22)
        + 0.20 * fit_to_target(row["active_ratio_env"], 0.72, 0.25)
    )

    text_risk = clamp(
        0.45 * row["text_difficulty"]
        + 0.20 * norm_range(row["speech_rate_wps"], 3.6, 5.0)
        + 0.20 * norm_range(row["zcr"], 0.11, 0.22)
        + 0.15 * norm_range(row["spectral_centroid_hz"], 950.0, 1650.0)
    )
    row["articulation_risk_inverse"] = clamp(1.0 - text_risk)
    row["delivery_fit_v1"] = clamp(
        0.36 * row["target_style_fit_v1"]
        + 0.22 * row["prosody_fit_light"]
        + 0.16 * row["pause_naturalness"]
        + 0.14 * row["spectral_balance_fit"]
        + 0.12 * row["articulation_risk_inverse"]
    )
```
