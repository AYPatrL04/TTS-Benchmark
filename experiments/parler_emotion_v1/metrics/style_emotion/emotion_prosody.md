# Emotion And Prosody Evaluation

- input: `experiments\parler_emotion_v1\inputs\parler_emotion_manifest.csv`
- model: `superb/wav2vec2-base-superb-er`
- samples: 8
- mean prosody activity: 0.731996
- mean target emotion probability: 0.530140
- top-label counts: {'happy': 4, 'angry': 1, 'neutral': 3}

Emotion is an utterance-level classifier proxy. Prosody activity combines pitch variance and energy dynamics. These are automatic style proxies, not final human preference.

| id | top_emotion | top_prob | target_prob | f0_std | energy_cv | prosody_activity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| happy_01 | happy | 0.745344 | 0.745344 | 49.567887 | 0.819194 | 0.696049 |
| happy_02 | happy | 0.961356 | 0.961356 | 53.650286 | 0.879178 | 0.724269 |
| sad_01 | happy | 0.917472 | 0.001039 | 58.657151 | 0.821375 | 0.767677 |
| sad_02 | happy | 0.546153 | 0.050295 | 58.770767 | 0.816529 | 0.768948 |
| angry_01 | angry | 0.431213 | 0.431213 | 49.804969 | 1.003768 | 0.755492 |
| angry_02 | neutral | 0.564551 | 0.083773 | 51.439394 | 0.955129 | 0.751986 |
| neutral_01 | neutral | 0.988873 | 0.988873 | 43.849048 | 0.814770 | 0.694447 |
| neutral_02 | neutral | 0.979231 | 0.979231 | 48.434644 | 0.760699 | 0.697101 |
