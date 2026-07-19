# Metric Design

## Output Contract

The benchmark is vector-first. Every valid sample reports:

```text
I: automatic intelligibility in [0,1]
E: robust target-emotion consensus in [0,1]
S: acoustic sanity in [0,1]
D: emotion-model disagreement in [0,1]
Main_auto_v3: provisional automatic scalar in [0,1]
ranking_eligible: hard screening status
```

Higher is better for `I`, `E`, `S`, and `Main_auto_v3`. Lower disagreement `D`
means the emotion evidence is more consistent. None of these values is a human
MOS or a human-confirmed emotion label.

## Intelligibility

Whisper `tiny.en` transcribes the generated audio after reference and hypothesis
normalization:

```text
WER = (substitutions + deletions + insertions) / reference_word_count
I   = clip(1 - WER, 0, 1)
```

CER is retained only as a diagnostic because it comes from the same transcript.

## Emotion Consensus

The emotion component uses target probabilities from emotion2vec and SUPERB,
plus a fixed-anchor interpretation of MSP-Dim arousal/dominance/valence:

```text
P_vad(e) = softmax_e(-0.5 * sum_k ((vad_k - anchor_e,k)/scale_k)^2)
E = median(P_emotion2vec(target), P_SUPERB(target), P_vad(target))
D = max(P_emotion2vec, P_SUPERB, P_vad)
  - min(P_emotion2vec, P_SUPERB, P_vad)
```

The VAD anchors are:

| emotion | arousal | dominance | valence |
| --- | ---: | ---: | ---: |
| neutral | 0.40 | 0.50 | 0.50 |
| happy | 0.70 | 0.70 | 0.75 |
| angry | 0.75 | 0.70 | 0.25 |
| sad | 0.30 | 0.35 | 0.30 |

The per-axis scales are `(0.25, 0.25, 0.30)`. The anchors are fixed design
hypotheses and are not fitted to the 52 evaluation clips.

## Acoustic Sanity

```text
penalty = 0.30*loudness_penalty
        + 0.30*silence_penalty
        + 0.20*clipping_penalty
        + 0.15*spectral_flatness_penalty
        + 0.05*short_duration_penalty
S = clip(1 - penalty, 0, 1)
```

This detects gross waveform failures only. It must not be named naturalness or
interpreted as predicted MOS.

## Composite

```text
Main_auto_v3 = I^0.55 * E^0.35 * S^0.10
ranking_eligible = (I >= 0.70) and (S >= 0.50)
```

The geometric form limits cross-component compensation. The exponents and
eligibility thresholds are provisional, automatically assessed design choices.
High `D` should be surfaced with the score and prevents a strong emotion claim.

## Diagnostics

CER, raw transcripts, all class probabilities, VAD, semitone F0 variation,
energy variation, voiced ratio, silence, speaking rate, and boundary type remain
separate fields. They are not silently compressed into the Main scalar.

## Missing Values

Missing required model output makes a sample invalid. It must not be converted
to zero. Aggregates must report valid coverage.

## Implementation

- Main and surrogate formulas: `scripts/analyze_automatic_emotion_consensus.py`
- Neural emotion inference: `scripts/evaluate_automatic_emotion_models.py`
- Full design and result comparison: `docs/main_and_surrogate_metric_comparison.md`
