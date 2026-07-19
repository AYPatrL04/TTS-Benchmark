# Parler Emotion TTS Generation

This run uses Parler-TTS with one fixed speaker description style per target emotion.

- model: `parler-tts/parler-tts-mini-v1`
- device: `cuda`
- dtype: `torch.float16`
- samples: 8
- seed: 2026
- temperature: 1.0000

| id | target_emotion | duration_sec | audio |
| --- | --- | ---: | --- |
| happy_01 | happy | 5.085 | `experiments\parler_emotion_v1\generated\parler_emotion\happy_01.wav` |
| happy_02 | happy | 4.551 | `experiments\parler_emotion_v1\generated\parler_emotion\happy_02.wav` |
| sad_01 | sad | 4.586 | `experiments\parler_emotion_v1\generated\parler_emotion\sad_01.wav` |
| sad_02 | sad | 5.050 | `experiments\parler_emotion_v1\generated\parler_emotion\sad_02.wav` |
| angry_01 | angry | 4.609 | `experiments\parler_emotion_v1\generated\parler_emotion\angry_01.wav` |
| angry_02 | angry | 4.783 | `experiments\parler_emotion_v1\generated\parler_emotion\angry_02.wav` |
| neutral_01 | neutral | 3.309 | `experiments\parler_emotion_v1\generated\parler_emotion\neutral_01.wav` |
| neutral_02 | neutral | 3.913 | `experiments\parler_emotion_v1\generated\parler_emotion\neutral_02.wav` |
