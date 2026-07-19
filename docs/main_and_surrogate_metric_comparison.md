# Main and Surrogate Metric Comparison

## Main Teacher

The current reference is `provisional_teacher_v2`:

```text
I = 1 - normalized_WER
E = target_emotion_prob
with learned MOS: teacher_v2 = 0.55*I + 0.35*E + 0.10*Q
without learned MOS: teacher_v2 = (0.55*I + 0.35*E) / 0.90
```

CER and robust semitone prosody features are diagnostics. Missing required
inputs invalidate a sample. The current 26 samples all use acoustic sanity as
the quality fallback, and 25/26 receive exactly 1.0; quality ranking is therefore
not represented in the current teacher data. Acoustic sanity is consequently a
diagnostic and eligibility constraint, not a positive scalar contribution.

## Surrogate Families

The low-DSP family uses waveform and manifest features without ASR or a neural
emotion model: rate, silence, energy variation, semitone/pitch proxies, pause
shape, spectral balance, voiced ratio, and text difficulty.

The medium-neural family adds one locally cached wav2vec2 SER encoder:

```text
neural target-emotion probability/logits
+ leave-one-out target-centroid embedding margin
+ pairwise embedding consistency
+ selected low-DSP features
```

Ridge coefficients are learned inside LOOCV folds. A leave-dataset-out check
holds out the regular or boundary subset. Because both subsets use Parler-TTS
and the same speaker, this is not held-out-system validation.

## Current Agreement

| Candidate | Validation | Pearson | Spearman | Kendall tau-b | Pairwise accuracy | MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SIM/SER + low-DSP | LOOCV | 0.956729 | 0.923419 | 0.778462 | 0.889231 | 0.030452 |
| SIM/SER + low-DSP | leave-dataset-out | 0.848784 | 0.822946 | 0.667731 | 0.830769 | 0.045037 |
| SER target probability only | direct | 0.888542 | 0.877607 | 0.747692 | 0.873846 | 0.258240 |
| best low-DSP ridge | LOOCV | 0.634198 | 0.540513 | 0.396923 | 0.698462 | 0.095619 |
| nested-selected low-DSP ridge | nested LOOCV | 0.585509 | 0.485812 | 0.316923 | 0.658462 | 0.106340 |
| raw embedding centroid | LOOCV | 0.072682 | 0.169231 | 0.120000 | 0.560000 | 0.135550 |

The strongest result is primarily teacher replication: the surrogate and main
teacher both depend on the same SER model family. It does not establish
agreement with human perception. Raw embedding similarity alone remains weak.

## Cost

Latest local timing on 26 cached clips with CUDA:

| Pipeline | Seconds/clip | Relative to v1 full teacher | Approx. speedup |
| --- | ---: | ---: | ---: |
| main pipeline (Whisper + SER + sanity) | 1.268069 | 1.000000 | 1.0x |
| low-DSP | 0.075700 | 0.059697 | 16.8x |
| SIM-like embedding only | 0.150019 | 0.118305 | 8.5x |
| SIM/SER + low-DSP | 0.179812 | 0.141799 | 7.1x |

These timings remain useful as engineering estimates, but must be remeasured
when ASR, MOS, or SER models change.

## Decision

Use low-DSP as a cheap failure detector and SIM/SER + low-DSP as an experimental
teacher-replication ranker. Route low-confidence or out-of-domain samples to the
full main pipeline. Do not use either surrogate as a final metric or reward.

The next defensible experiment needs grouped splits by TTS system and text,
Kendall/pairwise/worst-group reporting, and a completely unseen TTS system.
Human labels should become the calibration target; only after that should the
surrogate be fitted to a frozen main metric.
