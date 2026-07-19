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
| neutral_subtle | neutral | 4.807 | `experiments\automatic_emotion_consensus_v1\controlled_generation\audio\neutral_subtle.wav` |
| neutral_obvious | neutral | 4.690 | `experiments\automatic_emotion_consensus_v1\controlled_generation\audio\neutral_obvious.wav` |
| happy_subtle | happy | 4.098 | `experiments\automatic_emotion_consensus_v1\controlled_generation\audio\happy_subtle.wav` |
| happy_obvious | happy | 4.865 | `experiments\automatic_emotion_consensus_v1\controlled_generation\audio\happy_obvious.wav` |
| sad_subtle | sad | 5.341 | `experiments\automatic_emotion_consensus_v1\controlled_generation\audio\sad_subtle.wav` |
| sad_obvious | sad | 5.178 | `experiments\automatic_emotion_consensus_v1\controlled_generation\audio\sad_obvious.wav` |
| angry_subtle | angry | 4.400 | `experiments\automatic_emotion_consensus_v1\controlled_generation\audio\angry_subtle.wav` |
| angry_obvious | angry | 4.505 | `experiments\automatic_emotion_consensus_v1\controlled_generation\audio\angry_obvious.wav` |
