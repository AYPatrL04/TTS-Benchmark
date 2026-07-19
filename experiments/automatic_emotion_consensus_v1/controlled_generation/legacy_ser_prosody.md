# Emotion And Prosody Evaluation

- input: `experiments\automatic_emotion_consensus_v1\controlled_generation\generated_manifest.csv`
- model: `superb/wav2vec2-base-superb-er`
- samples: 8
- mean prosody activity: 0.742141
- mean target emotion probability: 0.353312
- top-label counts: {'neutral': 4, 'happy': 4}

Emotion is an utterance-level classifier proxy. Prosody activity combines pitch variance and energy dynamics. These are automatic style proxies, not final human preference.

| id | top_emotion | top_prob | target_prob | f0_std | energy_cv | prosody_activity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| neutral_subtle | neutral | 0.825173 | 0.825173 | 47.644385 | 0.830265 | 0.751739 |
| neutral_obvious | neutral | 0.724256 | 0.724256 | 43.246868 | 0.779181 | 0.685222 |
| happy_subtle | happy | 0.556490 | 0.556490 | 60.544756 | 0.838781 | 0.748189 |
| happy_obvious | happy | 0.691319 | 0.691319 | 61.350552 | 0.912313 | 0.774111 |
| sad_subtle | happy | 0.669125 | 0.015082 | 41.816481 | 0.986754 | 0.722158 |
| sad_obvious | neutral | 0.696944 | 0.005907 | 43.669296 | 0.942224 | 0.735070 |
| angry_subtle | neutral | 0.870051 | 0.005779 | 48.788467 | 0.793049 | 0.720433 |
| angry_obvious | happy | 0.908935 | 0.002492 | 67.398165 | 0.980177 | 0.800204 |
