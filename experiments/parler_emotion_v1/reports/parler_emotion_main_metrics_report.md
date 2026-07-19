# Main Metric Components V2 Report

This report joins three automatic main metric tracks:

- Intelligibility: Whisper-normalized ASR WER/CER.
- Acoustic sanity: clipping, silence, loudness, duration, and flatness checks. This is not naturalness or MOS.
- Style/emotion: SUPERB Wav2Vec2 emotion classifier plus pitch/energy prosody activity.

## Aggregate

- samples: 8
- mean WER: 0.026698
- mean CER: 0.007592
- mean intelligibility score: 97.330/100
- mean acoustic sanity: 1.000000
- mean prosody activity: 0.731996
- emotion top-label counts: {'happy': 4, 'angry': 1, 'neutral': 3}

## Worst WER

| id | WER | CER | naturalness | emotion | prosody | transcript |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| neutral_02 | 0.08333333 | 0.01470588 | 1.000000 | neutral | 0.697101 | Please open the Settings menu, choose Account Preferences and confirm they update. |
| sad_01 | 0.07142857 | 0.01785714 | 1.000000 | happy | 0.767677 | I'm sorry, I really miss the quiet mornings we use to share together. |
| angry_02 | 0.05882353 | 0.02816901 | 1.000000 | neutral | 0.751986 | No, the answer is not good enough. We have waited too long and I am extremely frustrated. |
| happy_01 | 0.00000000 | 0.00000000 | 1.000000 | happy | 0.696049 | I am so happy to see you today. This is wonderful news and I can hardly stop smiling. |
| happy_02 | 0.00000000 | 0.00000000 | 1.000000 | happy | 0.724269 | That was amazing. I feel excited, grateful, and full of energy right now. |

## Lowest Acoustic Sanity

| id | acoustic sanity | WER | silence_ratio | audio |
| --- | ---: | ---: | ---: | --- |
| happy_01 | 1.000000 | 0.00000000 | 0.18934911 | `experiments\parler_emotion_v1\generated\parler_emotion\happy_01.wav` |
| happy_02 | 1.000000 | 0.00000000 | 0.21145374 | `experiments\parler_emotion_v1\generated\parler_emotion\happy_02.wav` |
| sad_01 | 1.000000 | 0.07142857 | 0.15973742 | `experiments\parler_emotion_v1\generated\parler_emotion\sad_01.wav` |
| sad_02 | 1.000000 | 0.00000000 | 0.16269841 | `experiments\parler_emotion_v1\generated\parler_emotion\sad_02.wav` |
| angry_01 | 1.000000 | 0.00000000 | 0.25272331 | `experiments\parler_emotion_v1\generated\parler_emotion\angry_01.wav` |

## Lowest Style Proxy

| id | style_proxy | emotion | emotion_prob | prosody | audio |
| --- | ---: | --- | ---: | ---: | --- |
| sad_01 | 38.436 | happy | 0.917472 | 0.767677 | `experiments\parler_emotion_v1\generated\parler_emotion\sad_01.wav` |
| sad_02 | 40.962 | happy | 0.546153 | 0.768948 | `experiments\parler_emotion_v1\generated\parler_emotion\sad_02.wav` |
| angry_02 | 41.788 | neutral | 0.564551 | 0.751986 | `experiments\parler_emotion_v1\generated\parler_emotion\angry_02.wav` |
| angry_01 | 59.335 | angry | 0.431213 | 0.755492 | `experiments\parler_emotion_v1\generated\parler_emotion\angry_01.wav` |
| happy_01 | 72.070 | happy | 0.745344 | 0.696049 | `experiments\parler_emotion_v1\generated\parler_emotion\happy_01.wav` |

## Interpretation

- WER behaves as an intelligibility metric: substitutions, deletions, and insertions show where ASR could not recover the intended text.
- Acoustic sanity catches gross failures only. Add calibrated UTMOS and defect dimensions before interpreting Q as naturalness.
- Style/emotion results should be read as an automatic screening signal; SER can be sensitive to speaker identity, wording, and prosody artifacts.
- These automatic main metrics are suitable for screening and surrogate fitting, but still need a small human listening sanity check for boundary cases.
