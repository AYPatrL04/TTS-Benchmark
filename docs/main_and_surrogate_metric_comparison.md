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
