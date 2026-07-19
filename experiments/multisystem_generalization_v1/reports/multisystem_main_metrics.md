# Main Metric Components V2 Report

This report joins three automatic main metric tracks:

- Intelligibility: Whisper-normalized ASR WER/CER.
- Acoustic sanity: clipping, silence, loudness, duration, and flatness checks. This is not naturalness or MOS.
- Style/emotion: SUPERB Wav2Vec2 emotion classifier plus pitch/energy prosody activity.

## Aggregate

- samples: 18
- mean WER: 0.073972
- mean CER: 0.051104
- mean intelligibility score: 92.603/100
- mean acoustic sanity: 0.995527
- mean prosody activity: 0.729454
- emotion top-label counts: {'happy': 2, 'neutral': 12, 'angry': 4}

## Worst WER

| id | WER | CER | naturalness | emotion | prosody | transcript |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| parler__boundary_acronyms | 0.50000000 | 0.57142857 | 1.000000 | happy | 0.727245 | The 4-Shivroa uses Arca, Zinn and Yuzhil 8 in the final Cobo Kuiq release |
| bark__normal_01 | 0.20000000 | 0.18518519 | 1.000000 | neutral | 0.812187 | the final report before the meeting starts tomorrow. |
| parler__boundary_tongue_twister | 0.18181818 | 0.06349206 | 0.998833 | angry | 0.766458 | Redlar, yellow leather, unique New York, repeated twice without rushing |
| bark__boundary_tongue_twister | 0.18181818 | 0.07936508 | 0.920659 | neutral | 0.837990 | Red leather Yellow leather Unite New York Repeated twice without Russian |
| sapi__boundary_acronyms | 0.14285714 | 0.02040816 | 1.000000 | neutral | 0.651769 | The GPU API uses HD TPS, JSON, and UTF-8 in the final SDK release. |

## Lowest Acoustic Sanity

| id | acoustic sanity | WER | silence_ratio | audio |
| --- | ---: | ---: | ---: | --- |
| bark__boundary_tongue_twister | 0.920659 | 0.18181818 | 0.54713115 | `experiments\multisystem_generalization_v1\generated\bark_speaker_6\boundary_tongue_twister.wav` |
| parler__boundary_tongue_twister | 0.998833 | 0.18181818 | 0.23185012 | `experiments\multisystem_generalization_v1\generated\parler_jenna\boundary_tongue_twister.wav` |
| parler__normal_01 | 1.000000 | 0.00000000 | 0.11695906 | `experiments\multisystem_generalization_v1\generated\parler_jenna\normal_01.wav` |
| parler__normal_02 | 1.000000 | 0.00000000 | 0.10271903 | `experiments\multisystem_generalization_v1\generated\parler_jenna\normal_02.wav` |
| parler__normal_03 | 1.000000 | 0.00000000 | 0.13492063 | `experiments\multisystem_generalization_v1\generated\parler_jenna\normal_03.wav` |

## Lowest Style Proxy

| id | style_proxy | emotion | emotion_prob | prosody | audio |
| --- | ---: | --- | ---: | ---: | --- |
| parler__boundary_tongue_twister | 45.783 | angry | 0.490146 | 0.766458 | `experiments\multisystem_generalization_v1\generated\parler_jenna\boundary_tongue_twister.wav` |
| sapi__normal_01 | 47.018 | angry | 0.690974 | 0.677561 | `experiments\multisystem_generalization_v1\generated\sapi_zira\normal_01.wav` |
| sapi__boundary_tongue_twister | 48.193 | angry | 0.510925 | 0.687532 | `experiments\multisystem_generalization_v1\generated\sapi_zira\boundary_tongue_twister.wav` |
| sapi__normal_03 | 49.570 | angry | 0.575699 | 0.645481 | `experiments\multisystem_generalization_v1\generated\sapi_zira\normal_03.wav` |
| parler__boundary_acronyms | 52.995 | happy | 0.647478 | 0.727245 | `experiments\multisystem_generalization_v1\generated\parler_jenna\boundary_acronyms.wav` |

## Interpretation

- WER behaves as an intelligibility metric: substitutions, deletions, and insertions show where ASR could not recover the intended text.
- Acoustic sanity catches gross failures only. Add calibrated UTMOS and defect dimensions before interpreting Q as naturalness.
- Style/emotion results should be read as an automatic screening signal; SER can be sensitive to speaker identity, wording, and prosody artifacts.
- These automatic main metrics are suitable for screening and surrogate fitting, but still need a small human listening sanity check for boundary cases.
