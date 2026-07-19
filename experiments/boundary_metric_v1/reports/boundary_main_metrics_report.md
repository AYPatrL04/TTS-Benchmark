# Main Metrics V1 Report

This report joins three automatic main metric tracks:

- Intelligibility: Whisper-normalized ASR WER/CER.
- Naturalness: lightweight acoustic naturalness proxy fallback. UTMOS script is implemented separately but the weight download did not complete in this run.
- Style/emotion: SUPERB Wav2Vec2 emotion classifier plus pitch/energy prosody activity.

## Aggregate

- samples: 18
- mean WER: 0.074980
- mean CER: 0.051544
- mean intelligibility score: 92.502/100
- mean naturalness proxy: 4.999764/5
- mean prosody activity: 0.934196
- emotion top-label counts: {'neutral': 12, 'happy': 4, 'angry': 2}

## Worst WER

| id | WER | CER | naturalness | emotion | prosody | transcript |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| fast_tongue_twister | 0.50000000 | 0.16129032 | 5.000000 | neutral | 1.000000 | She sells seashells by the seashore then swiftly switches six six groups. |
| technical_acronyms | 0.38461538 | 0.51923077 | 5.000000 | neutral | 0.954739 | the pocha loads jump from dropping, then sends twice requests to the charter. |
| neutral_text_angry_voice | 0.08333333 | 0.02000000 | 5.000000 | neutral | 1.000000 | The package arrived at the front desk at 3.15 in the afternoon. |
| noisy_neutral | 0.08333333 | 0.02000000 | 5.000000 | neutral | 0.929980 | The package arrived at the front desk at 3.15 in the afternoon. |
| distant_reverb_neutral | 0.08333333 | 0.07352941 | 4.995757 | neutral | 0.951073 | Please open the settings menu, choose account preferences, and confirm the line. |

## Lowest Naturalness Proxy

| id | naturalness | WER | silence_ratio | audio |
| --- | ---: | ---: | ---: | --- |
| distant_reverb_neutral | 4.995757 | 0.08333333 | 0.20851064 | `experiments\boundary_metric_v1\generated\parler_boundary\distant_reverb_neutral.wav` |
| control_happy_happy | 5.000000 | 0.00000000 | 0.24338624 | `experiments\boundary_metric_v1\generated\parler_boundary\control_happy_happy.wav` |
| control_sad_sad | 5.000000 | 0.00000000 | 0.19829424 | `experiments\boundary_metric_v1\generated\parler_boundary\control_sad_sad.wav` |
| control_angry_angry | 5.000000 | 0.00000000 | 0.21000000 | `experiments\boundary_metric_v1\generated\parler_boundary\control_angry_angry.wav` |
| control_neutral_neutral | 5.000000 | 0.07142857 | 0.20135747 | `experiments\boundary_metric_v1\generated\parler_boundary\control_neutral_neutral.wav` |

## Lowest Style Proxy

| id | style_proxy | emotion | emotion_prob | prosody | audio |
| --- | ---: | --- | ---: | ---: | --- |
| whisper_sad | 46.694 | angry | 0.925998 | 0.933154 | `experiments\boundary_metric_v1\generated\parler_boundary\whisper_sad.wav` |
| neutral_text_angry_voice | 50.106 | neutral | 0.857578 | 1.000000 | `experiments\boundary_metric_v1\generated\parler_boundary\neutral_text_angry_voice.wav` |
| control_sad_sad | 50.127 | happy | 0.980182 | 1.000000 | `experiments\boundary_metric_v1\generated\parler_boundary\control_sad_sad.wav` |
| happy_text_sad_voice | 51.009 | happy | 0.953662 | 1.000000 | `experiments\boundary_metric_v1\generated\parler_boundary\happy_text_sad_voice.wav` |
| control_happy_happy | 55.199 | neutral | 0.470497 | 1.000000 | `experiments\boundary_metric_v1\generated\parler_boundary\control_happy_happy.wav` |

## Interpretation

- WER behaves as an intelligibility metric: substitutions, deletions, and insertions show where ASR could not recover the intended text.
- The naturalness fallback mostly catches gross acoustic issues, not semantic omissions. Replace it with UTMOS/NISQA when weights are available.
- Style/emotion results should be read as an automatic screening signal; SER can be sensitive to speaker identity, wording, and prosody artifacts.
- These automatic main metrics are suitable for screening and surrogate fitting, but still need a small human listening sanity check for boundary cases.
