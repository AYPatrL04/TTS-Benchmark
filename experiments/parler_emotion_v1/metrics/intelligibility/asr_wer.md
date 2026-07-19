# ASR-WER Evaluation

- input: `experiments\parler_emotion_v1\inputs\parler_emotion_manifest.csv`
- ASR model: `openai/whisper-tiny.en`
- samples: 8
- mean WER: 0.02669818
- mean CER: 0.00759150

WER is computed by transcribing generated audio with ASR, normalizing text, then comparing the ASR transcript against the input text.
This is an automatic intelligibility proxy, not a replacement for small human listening checks.

| id | WER | CER | S/D/I | transcript |
| --- | ---: | ---: | --- | --- |
| happy_01 | 0.00000000 | 0.00000000 | 0/0/0 | I am so happy to see you today. This is wonderful news and I can hardly stop smiling. |
| happy_02 | 0.00000000 | 0.00000000 | 0/0/0 | That was amazing. I feel excited, grateful, and full of energy right now. |
| sad_02 | 0.00000000 | 0.00000000 | 0/0/0 | The room feels empty today, and every small sound reminds me that you're gone. |
| angry_01 | 0.00000000 | 0.00000000 | 0/0/0 | I told you not to touch that file, this is unacceptable, and I need you to fix it now. |
| neutral_01 | 0.00000000 | 0.00000000 | 0/0/0 | The package arrived at the front desk at 315 in the afternoon. |
| angry_02 | 0.05882353 | 0.02816901 | 1/0/0 | No, the answer is not good enough. We have waited too long and I am extremely frustrated. |
| sad_01 | 0.07142857 | 0.01785714 | 1/0/0 | I'm sorry, I really miss the quiet mornings we use to share together. |
| neutral_02 | 0.08333333 | 0.01470588 | 1/0/0 | Please open the Settings menu, choose Account Preferences and confirm they update. |
