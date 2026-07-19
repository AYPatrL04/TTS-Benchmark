# Main Metric Design

Current version: 2026-07-19

## Goal

Define a single automatic main metric for emotion-aware TTS evaluation.

The score should measure whether generated speech:

- preserves the input text;
- sounds acoustically usable;
- expresses the intended emotion;
- uses prosody that fits the intended emotion.

All component scores and the final score are normalized to `0-1`, where higher
is better.

This main metric is intended to be the reference target for later surrogate
metric search. Surrogate metrics should be selected by how well they correlate
with and rank against this score.

## Required Inputs

The scoring pipeline expects a combined CSV with these fields:

| field | meaning |
| --- | --- |
| `id` | sample id |
| `text` | intended input text |
| `audio_path` | generated audio path |
| `target_emotion` | intended emotion label |
| `wer` | ASR word error rate |
| `cer` | ASR character error rate |
| `naturalness_proxy_1_5` | acoustic quality proxy on a 1-5 scale |
| `target_emotion_prob` | SER probability assigned to the target emotion |
| `target_emotion_match` | `1` if SER top label equals target emotion, else `0` |
| `prosody_activity_0_1` | pitch/energy activity score |

Implementation:

```text
scripts/score_emotion_tts_main_metric.py
```

## Formula

```text
I = 0.80 * (1 - WER) + 0.20 * (1 - CER)
Q = (naturalness_proxy_1_5 - 1) / 4
E = 0.70 * target_emotion_prob + 0.30 * target_emotion_match
P = 1 - abs(prosody_activity - target_prosody) / tolerance

raw = 0.45 * I + 0.15 * Q + 0.30 * E + 0.10 * P
gate = 0.35 + 0.65 * I

main_metric = raw * gate
```

All intermediate values are clipped to `[0, 1]` when needed.

## Components

### Intelligibility `I`

```text
I = 0.80 * (1 - WER) + 0.20 * (1 - CER)
```

`I` measures whether the generated audio preserves the intended text.

WER receives the larger weight because word-level errors are more damaging for
TTS usefulness. CER is included to soften cases where word tokenization,
numbers, punctuation, or spelling variants make WER too harsh.

### Acoustic Quality `Q`

```text
Q = (naturalness_proxy_1_5 - 1) / 4
```

`Q` converts a `1-5` naturalness proxy into the shared `0-1` scale.

The current proxy is a lightweight no-reference acoustic score. It is useful as
a basic quality guard, but should not be treated as a final perceptual MOS
replacement.

### Emotion Match `E`

```text
E = 0.70 * target_emotion_prob + 0.30 * target_emotion_match
```

`E` measures whether the generated audio expresses the intended emotion.

The target emotion probability captures confidence. The top-label match adds a
categorical reward when the predicted emotion is exactly the intended one.

### Prosody Fit `P`

```text
P = 1 - abs(prosody_activity - target_prosody) / tolerance
```

`P` measures whether pitch and energy movement are appropriate for the intended
emotion. It is target-dependent: stronger prosody is not always better.

Current target settings:

| emotion | target prosody activity | tolerance |
| --- | ---: | ---: |
| happy | 0.85 | 0.35 |
| angry | 0.90 | 0.35 |
| sad | 0.60 | 0.40 |
| neutral | 0.80 | 0.40 |

## Weights

| component | weight | rationale |
| --- | ---: | --- |
| `I` intelligibility | 0.45 | text preservation is the primary requirement |
| `E` emotion match | 0.30 | emotion control is central to this task |
| `Q` acoustic quality | 0.15 | audio quality matters, but the current proxy is still coarse |
| `P` prosody fit | 0.10 | prosody helps style evaluation but is not enough alone |

The intelligibility gate is:

```text
gate = 0.35 + 0.65 * I
```

This prevents samples with poor text preservation from receiving high final
scores only because emotion or acoustic signals look good.

## Interpretation

Suggested score bands:

| score range | interpretation |
| --- | --- |
| `0.90-1.00` | strong automatic pass |
| `0.75-0.90` | generally acceptable, inspect if used as a top candidate |
| `0.60-0.75` | mixed quality or style mismatch likely |
| `<0.60` | likely failure or strong metric disagreement |

These bands are practical screening bands, not human MOS labels.

## Output Fields

The scoring script writes:

| field | meaning |
| --- | --- |
| `main_metric_0_1` | final gated composite score |
| `main_metric_raw_0_1` | weighted score before intelligibility gate |
| `intelligibility_component_0_1` | `I` |
| `naturalness_component_0_1` | `Q` |
| `emotion_component_0_1` | `E` |
| `prosody_fit_component_0_1` | `P` |
| `intelligibility_gate_0_1` | gate multiplier |

## Usage

```powershell
conda run -n TTS python scripts\score_emotion_tts_main_metric.py `
  --input path\to\combined_metrics.csv `
  --output-csv path\to\scored_main_metric.csv `
  --output-md path\to\scored_main_metric.md `
  --experiment-name ExperimentName
```

## Limitations

This main metric is automatic and should be treated as a benchmark target for
surrogate discovery, not as perfect human judgment.

Known limitations:

- ASR-based WER/CER can be sensitive to numbers, acronyms, punctuation, and
  normalization choices.
- Single-model emotion classification can misread style, speaker traits, or
  lexical content.
- The current naturalness proxy is too coarse for subtle perceptual quality.
- Prosody activity measures pitch/energy movement, not semantic
  appropriateness by itself.

## Improvement Directions

Near-term improvements:

- use a stronger ASR model and stricter text normalization;
- replace the lightweight naturalness proxy with UTMOS, NISQA, DNSMOS, or a
  comparable MOS predictor;
- add a second independent emotion/style model and average or calibrate the
  emotion score;
- collect a small human AB/MOS calibration set for boundary cases;
- fit cheap surrogate metrics against this main metric after the main metric is
  fixed.
