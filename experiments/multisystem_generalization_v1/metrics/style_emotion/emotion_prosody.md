# Emotion And Prosody Evaluation

- input: `experiments\multisystem_generalization_v1\inputs\multisystem_manifest.csv`
- model: `superb/wav2vec2-base-superb-er`
- samples: 18
- mean prosody activity: 0.729454
- mean target emotion probability: 0.672620
- top-label counts: {'happy': 2, 'neutral': 12, 'angry': 4}

Emotion is an utterance-level classifier proxy. Prosody activity combines pitch variance and energy dynamics. These are automatic style proxies, not final human preference.

| id | top_emotion | top_prob | target_prob | f0_std | energy_cv | prosody_activity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| parler__normal_01 | happy | 0.456406 | 0.425097 | 61.697725 | 0.828984 | 0.752040 |
| parler__normal_02 | neutral | 0.739594 | 0.739594 | 54.071328 | 0.676603 | 0.699519 |
| parler__normal_03 | neutral | 0.694727 | 0.694727 | 40.699825 | 0.830960 | 0.701526 |
| parler__boundary_digits | neutral | 0.936503 | 0.936503 | 39.245784 | 0.734957 | 0.660141 |
| parler__boundary_acronyms | happy | 0.647478 | 0.332647 | 54.938705 | 0.864839 | 0.727245 |
| parler__boundary_tongue_twister | angry | 0.490146 | 0.149201 | 55.026446 | 0.951753 | 0.766458 |
| bark__normal_01 | neutral | 0.944272 | 0.944272 | 47.973866 | 1.094057 | 0.812187 |
| bark__normal_02 | neutral | 0.998047 | 0.998047 | 33.058819 | 1.279638 | 0.777517 |
| bark__normal_03 | neutral | 0.853051 | 0.853051 | 48.205283 | 0.858321 | 0.769735 |
| bark__boundary_digits | neutral | 0.984037 | 0.984037 | 44.669503 | 1.063258 | 0.801795 |
| bark__boundary_acronyms | neutral | 0.998164 | 0.998164 | 71.305420 | 1.197806 | 0.867451 |
| bark__boundary_tongue_twister | neutral | 0.994678 | 0.994678 | 43.821779 | 1.416172 | 0.837990 |
| sapi__normal_01 | angry | 0.690974 | 0.262801 | 30.973134 | 0.891997 | 0.677561 |
| sapi__normal_02 | neutral | 0.768729 | 0.768729 | 25.289323 | 1.005243 | 0.674410 |
| sapi__normal_03 | angry | 0.575699 | 0.345919 | 23.769096 | 0.926163 | 0.645481 |
| sapi__boundary_digits | neutral | 0.586868 | 0.586868 | 23.074830 | 0.831172 | 0.619814 |
| sapi__boundary_acronyms | neutral | 0.816497 | 0.816497 | 34.490251 | 0.772002 | 0.651769 |
| sapi__boundary_tongue_twister | angry | 0.510925 | 0.276323 | 26.286901 | 1.041880 | 0.687532 |
