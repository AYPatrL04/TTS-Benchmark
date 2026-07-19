# Main Metric Components V2 Report

This report joins three automatic main metric tracks:

- Intelligibility: Whisper-normalized ASR WER/CER.
- Acoustic sanity: clipping, silence, loudness, duration, and flatness checks. This is not naturalness or MOS.
- Style/emotion: SUPERB Wav2Vec2 emotion classifier plus pitch/energy prosody activity.

## Aggregate

- samples: 18
- mean WER: 0.074980
- mean CER: 0.051544
- mean intelligibility score: 92.502/100
- mean acoustic sanity: 0.999941
- mean prosody activity: 0.740239
- emotion top-label counts: {'neutral': 12, 'happy': 4, 'angry': 2}

## Worst WER

| id | WER | CER | naturalness | emotion | prosody | transcript |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| fast_tongue_twister | 0.50000000 | 0.16129032 | 1.000000 | neutral | 0.763100 | She sells seashells by the seashore then swiftly switches six six groups. |
| technical_acronyms | 0.38461538 | 0.51923077 | 1.000000 | neutral | 0.724991 | the pocha loads jump from dropping, then sends twice requests to the charter. |
| neutral_text_angry_voice | 0.08333333 | 0.02000000 | 1.000000 | neutral | 0.771837 | The package arrived at the front desk at 3.15 in the afternoon. |
| noisy_neutral | 0.08333333 | 0.02000000 | 1.000000 | neutral | 0.733051 | The package arrived at the front desk at 3.15 in the afternoon. |
| distant_reverb_neutral | 0.08333333 | 0.07352941 | 0.998939 | neutral | 0.721494 | Please open the settings menu, choose account preferences, and confirm the line. |

## Lowest Acoustic Sanity

| id | acoustic sanity | WER | silence_ratio | audio |
| --- | ---: | ---: | ---: | --- |
| distant_reverb_neutral | 0.998939 | 0.08333333 | 0.20851064 | `experiments\boundary_metric_v1\generated\parler_boundary\distant_reverb_neutral.wav` |
| control_happy_happy | 1.000000 | 0.00000000 | 0.24338624 | `experiments\boundary_metric_v1\generated\parler_boundary\control_happy_happy.wav` |
| control_sad_sad | 1.000000 | 0.00000000 | 0.19829424 | `experiments\boundary_metric_v1\generated\parler_boundary\control_sad_sad.wav` |
| control_angry_angry | 1.000000 | 0.00000000 | 0.21000000 | `experiments\boundary_metric_v1\generated\parler_boundary\control_angry_angry.wav` |
| control_neutral_neutral | 1.000000 | 0.07142857 | 0.20135747 | `experiments\boundary_metric_v1\generated\parler_boundary\control_neutral_neutral.wav` |

## Lowest Style Proxy

| id | style_proxy | emotion | emotion_prob | prosody | audio |
| --- | ---: | --- | ---: | ---: | --- |
| whisper_sad | 37.738 | angry | 0.925998 | 0.754031 | `experiments\boundary_metric_v1\generated\parler_boundary\whisper_sad.wav` |
| happy_text_sad_voice | 38.389 | happy | 0.953662 | 0.747598 | `experiments\boundary_metric_v1\generated\parler_boundary\happy_text_sad_voice.wav` |
| neutral_text_angry_voice | 38.697 | neutral | 0.857578 | 0.771837 | `experiments\boundary_metric_v1\generated\parler_boundary\neutral_text_angry_voice.wav` |
| control_sad_sad | 39.308 | happy | 0.980182 | 0.783611 | `experiments\boundary_metric_v1\generated\parler_boundary\control_sad_sad.wav` |
| control_happy_happy | 44.478 | neutral | 0.470497 | 0.785590 | `experiments\boundary_metric_v1\generated\parler_boundary\control_happy_happy.wav` |

## Interpretation

- WER behaves as an intelligibility metric: substitutions, deletions, and insertions show where ASR could not recover the intended text.
- Acoustic sanity catches gross failures only. Add calibrated UTMOS and defect dimensions before interpreting Q as naturalness.
- Style/emotion results should be read as an automatic screening signal; SER can be sensitive to speaker identity, wording, and prosody artifacts.
- These automatic main metrics are suitable for screening and surrogate fitting, but still need a small human listening sanity check for boundary cases.
