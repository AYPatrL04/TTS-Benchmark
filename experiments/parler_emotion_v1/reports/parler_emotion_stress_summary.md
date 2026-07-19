# Emotion Stress Main Metrics

- input: `experiments\parler_emotion_v1\combined\parler_emotion_main_metrics.csv`
- samples: 8

This summarizes the three automatic main metric tracks by intended text emotion.

## By Target Emotion

| target | n | mean WER | mean naturalness | mean target emotion prob | match rate | mean prosody |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| angry | 2 | 0.029412 | 5.000000 | 0.257493 | 0.500000 | 1.000000 |
| happy | 2 | 0.000000 | 5.000000 | 0.853350 | 1.000000 | 0.971770 |
| neutral | 2 | 0.041667 | 5.000000 | 0.984052 | 1.000000 | 0.931236 |
| sad | 2 | 0.035714 | 5.000000 | 0.025667 | 0.000000 | 0.954973 |

## Per Sample

| id | target | WER | naturalness | predicted emotion | target prob | match | prosody | transcript |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| happy_01 | happy | 0.00000000 | 5.000000 | happy | 0.745344 | 1 | 0.955108 | I am so happy to see you today. This is wonderful news and I can hardly stop smiling. |
| happy_02 | happy | 0.00000000 | 5.000000 | happy | 0.961356 | 1 | 0.988432 | That was amazing. I feel excited, grateful, and full of energy right now. |
| sad_01 | sad | 0.07142857 | 5.000000 | happy | 0.001039 | 0 | 0.956320 | I'm sorry, I really miss the quiet mornings we use to share together. |
| sad_02 | sad | 0.00000000 | 5.000000 | happy | 0.050295 | 0 | 0.953627 | The room feels empty today, and every small sound reminds me that you're gone. |
| angry_01 | angry | 0.00000000 | 5.000000 | angry | 0.431213 | 1 | 1.000000 | I told you not to touch that file, this is unacceptable, and I need you to fix it now. |
| angry_02 | angry | 0.05882353 | 5.000000 | neutral | 0.083773 | 0 | 1.000000 | No, the answer is not good enough. We have waited too long and I am extremely frustrated. |
| neutral_01 | neutral | 0.00000000 | 5.000000 | neutral | 0.988873 | 1 | 0.939862 | The package arrived at the front desk at 315 in the afternoon. |
| neutral_02 | neutral | 0.08333333 | 5.000000 | neutral | 0.979231 | 1 | 0.922610 | Please open the Settings menu, choose Account Preferences and confirm they update. |

## Readout

- WER/CER measure text intelligibility and should be interpreted separately from style or emotion control.
- Target emotion probability and match rate are automatic classifier checks, not human ground truth.
- High prosody activity means the audio has pitch/energy variation; it does not prove that the intended emotion was expressed.
- A flat naturalness proxy means this fallback metric is mostly catching gross acoustic defects, not subtle emotional quality.
- Low style rows should be prioritized for listening checks because automatic SER models can confuse affect, speaker traits, and lexical content.
