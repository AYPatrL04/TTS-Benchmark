# Metric Design

## Output Contract

The benchmark is vector-first. Every valid sample reports:

```text
(I, Q, E, prosody diagnostics, acoustic sanity flags)
```

`main_metric_0_1` is an experimental scalar named
`provisional_teacher_v2`. It is useful for pipeline and surrogate experiments,
but is not a human-calibrated ground truth, MOS, or publication-grade model
ranking metric.

Missing required values produce `metric_status=invalid`; missing values are
never converted to zero. Reports include valid-sample coverage.

## Components

### Intelligibility

```text
I = clip(1 - normalized_WER, 0, 1)
```

WER is computed from a Whisper transcript after text normalization. CER remains
in `cer_diagnostic_0_1`, but is not blended with WER because both derive from
the same ASR transcript. The output also retains transcript and word edits for
error analysis.

This measures TTS errors, ASR errors, and normalization errors together. The
current `whisper-tiny.en` result therefore needs stronger or heterogeneous ASR
validation before cross-system use.

### Quality and Acoustic Sanity

If a learned MOS column is present:

```text
Q = clip((utmos_score_1_5 - 1) / 4, 0, 1)
quality_component_source = learned_mos
```

Without learned MOS:

```text
quality_component_source = acoustic_sanity_fallback
```

The fallback checks gross loudness, silence, clipping, duration, and spectral
flatness failures. It is deliberately named acoustic sanity, not naturalness.
It cannot rank robotic speech, vocoder artifacts, reverberation, or subtle
prosody problems reliably. It is reported and used by `ranking_eligible`, but
does not add positive weight to the scalar.

### Emotion

```text
E = clip(target_emotion_prob, 0, 1)
```

The previous hard argmax-match bonus was removed because it duplicated the same
classifier evidence and introduced a discontinuous score jump. The current
SUPERB model is an uncalibrated IEMOCAP SER teacher applied out of domain to
synthetic speech. Its probability is retained for experimentation, not treated
as human emotion agreement.

### Prosody

Prosody is diagnostic and excluded from the scalar. Pitch variation is measured
in semitones relative to the utterance median, with standard deviation, median
absolute deviation, and p90-p10 range. Energy variation, voiced ratio, speaking
rate, and pause behavior remain separate features.

This replaces speaker-dependent Hz-only activity as the preferred description.
Emotion targets must eventually be learned as conditional distributions from
human speech instead of hand-set point targets.

## Provisional Scalar

```text
with learned MOS:
teacher_v2 = clip(0.55 * I + 0.35 * E + 0.10 * Q, 0, 1)

without learned MOS:
teacher_v2 = clip((0.55 * I + 0.35 * E) / 0.90, 0, 1)

ranking_eligible = (I >= 0.70) and (acoustic_sanity >= 0.50)
```

The weights are not human-calibrated. `ranking_eligible` is a hard screening
constraint, not a claim that 0.70 is a perceptual acceptability threshold.
Where different failure modes matter, compare the component vector or Pareto
front rather than relying on the scalar.

## Output Fields

| Field | Meaning |
| --- | --- |
| `metric_version` | `provisional_teacher_v2` |
| `metric_status` | `valid_provisional` or `invalid` |
| `metric_missing_fields` | missing required inputs |
| `main_metric_0_1` | provisional teacher scalar |
| `ranking_eligible` | hard intelligibility/sanity screen |
| `intelligibility_component_0_1` | `I` |
| `quality_component_0_1` | learned MOS or sanity diagnostic |
| `quality_component_source` | learned MOS or sanity fallback |
| `emotion_component_0_1` | `E` |
| `cer_diagnostic_0_1` | CER, not in scalar |
| `prosody_diagnostic_0_1` | activity descriptor, not in scalar |
| `teacher_active_weights` | weights used for the current row |

## Validation Required

The current data contain 26 clips from one Parler-TTS configuration and largely
one speaker. Before using the metric for model ranking or reward learning:

1. Collect human intelligibility, naturalness, emotion-match, and overall
   preference labels with multiple raters.
2. Add structurally different TTS systems, speakers, texts, natural failures,
   and controlled degradations.
3. Calibrate ASR, MOS, and SER outputs on synthetic speech.
4. Learn a non-negative monotonic composite mapping from human judgments.
5. Hold out complete TTS systems and texts for final evaluation.

Likely quality extensions are calibrated UTMOS plus defect dimensions from a
model such as NISQA. Likely emotion extensions are heterogeneous SER models and
continuous valence-arousal predictions. These are future integrations, not
current measured components.
