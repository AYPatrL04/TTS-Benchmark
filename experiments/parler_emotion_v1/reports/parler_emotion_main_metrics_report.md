# Main Metrics V1 Report

This report joins three automatic main metric tracks:

- Intelligibility: Whisper-normalized ASR WER/CER.
- Naturalness: lightweight acoustic naturalness proxy fallback. UTMOS script is implemented separately but the weight download did not complete in this run.
- Style/emotion: SUPERB Wav2Vec2 emotion classifier plus pitch/energy prosody activity.

## Aggregate

- samples: 8
- mean WER: 0.026698
- mean CER: 0.007592
- mean intelligibility score: 97.330/100
- mean naturalness proxy: 5.000000/5
- mean prosody activity: 0.964495
- emotion top-label counts: {'happy': 4, 'angry': 1, 'neutral': 3}

## Worst WER

| id | WER | CER | naturalness | emotion | prosody | transcript |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| neutral_02 | 0.08333333 | 0.01470588 | 5.000000 | neutral | 0.922610 | Please open the Settings menu, choose Account Preferences and confirm they update. |
| sad_01 | 0.07142857 | 0.01785714 | 5.000000 | happy | 0.956320 | I'm sorry, I really miss the quiet mornings we use to share together. |
| angry_02 | 0.05882353 | 0.02816901 | 5.000000 | neutral | 1.000000 | No, the answer is not good enough. We have waited too long and I am extremely frustrated. |
| happy_01 | 0.00000000 | 0.00000000 | 5.000000 | happy | 0.955108 | I am so happy to see you today. This is wonderful news and I can hardly stop smiling. |
| happy_02 | 0.00000000 | 0.00000000 | 5.000000 | happy | 0.988432 | That was amazing. I feel excited, grateful, and full of energy right now. |

## Lowest Naturalness Proxy

| id | naturalness | WER | silence_ratio | audio |
| --- | ---: | ---: | ---: | --- |
| happy_01 | 5.000000 | 0.00000000 | 0.18934911 | `experiments\parler_emotion_v1\generated\parler_emotion\happy_01.wav` |
| happy_02 | 5.000000 | 0.00000000 | 0.21145374 | `experiments\parler_emotion_v1\generated\parler_emotion\happy_02.wav` |
| sad_01 | 5.000000 | 0.07142857 | 0.15973742 | `experiments\parler_emotion_v1\generated\parler_emotion\sad_01.wav` |
| sad_02 | 5.000000 | 0.00000000 | 0.16269841 | `experiments\parler_emotion_v1\generated\parler_emotion\sad_02.wav` |
| angry_01 | 5.000000 | 0.00000000 | 0.25272331 | `experiments\parler_emotion_v1\generated\parler_emotion\angry_01.wav` |

## Lowest Style Proxy

| id | style_proxy | emotion | emotion_prob | prosody | audio |
| --- | ---: | --- | ---: | ---: | --- |
| sad_01 | 47.868 | happy | 0.917472 | 0.956320 | `experiments\parler_emotion_v1\generated\parler_emotion\sad_01.wav` |
| sad_02 | 50.196 | happy | 0.546153 | 0.953627 | `experiments\parler_emotion_v1\generated\parler_emotion\sad_02.wav` |
| angry_02 | 54.189 | neutral | 0.564551 | 1.000000 | `experiments\parler_emotion_v1\generated\parler_emotion\angry_02.wav` |
| angry_01 | 71.561 | angry | 0.431213 | 1.000000 | `experiments\parler_emotion_v1\generated\parler_emotion\angry_01.wav` |
| happy_01 | 85.023 | happy | 0.745344 | 0.955108 | `experiments\parler_emotion_v1\generated\parler_emotion\happy_01.wav` |

## Interpretation

- WER behaves as an intelligibility metric: substitutions, deletions, and insertions show where ASR could not recover the intended text.
- The naturalness fallback mostly catches gross acoustic issues, not semantic omissions. Replace it with UTMOS/NISQA when weights are available.
- Style/emotion results should be read as an automatic screening signal; SER can be sensitive to speaker identity, wording, and prosody artifacts.
- These automatic main metrics are suitable for screening and surrogate fitting, but still need a small human listening sanity check for boundary cases.
