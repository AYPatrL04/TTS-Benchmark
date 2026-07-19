# Emotion And Prosody Evaluation

- input: `experiments\boundary_metric_v1\inputs\boundary_manifest.csv`
- model: `superb/wav2vec2-base-superb-er`
- samples: 18
- mean prosody activity: 0.740239
- mean target emotion probability: 0.559545
- top-label counts: {'neutral': 12, 'happy': 4, 'angry': 2}

Emotion is an utterance-level classifier proxy. Prosody activity combines pitch variance and energy dynamics. These are automatic style proxies, not final human preference.

| id | top_emotion | top_prob | target_prob | f0_std | energy_cv | prosody_activity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| control_happy_happy | neutral | 0.470497 | 0.103971 | 65.928099 | 0.901445 | 0.785590 |
| control_sad_sad | happy | 0.980182 | 0.002545 | 59.720599 | 0.925057 | 0.783611 |
| control_angry_angry | angry | 0.350994 | 0.350994 | 56.700273 | 0.983540 | 0.788443 |
| control_neutral_neutral | neutral | 0.795235 | 0.795235 | 62.397275 | 0.870611 | 0.766453 |
| happy_text_sad_voice | happy | 0.953662 | 0.020172 | 54.878979 | 0.908329 | 0.747598 |
| sad_text_happy_voice | happy | 0.936487 | 0.936487 | 66.857580 | 0.759073 | 0.747225 |
| angry_text_neutral_voice | neutral | 0.756618 | 0.756618 | 67.037091 | 0.831952 | 0.792187 |
| neutral_text_angry_voice | neutral | 0.857578 | 0.002112 | 63.891080 | 0.957947 | 0.771837 |
| digits_address | neutral | 0.919919 | 0.919919 | 62.566842 | 0.781618 | 0.736143 |
| homophones_minimal_pairs | neutral | 0.748683 | 0.748683 | 39.204922 | 0.695518 | 0.662124 |
| function_word_repetition | neutral | 0.976893 | 0.976893 | 30.183245 | 0.823153 | 0.650205 |
| technical_acronyms | neutral | 0.886291 | 0.886291 | 59.106612 | 0.818530 | 0.724991 |
| fast_tongue_twister | neutral | 0.598071 | 0.598071 | 50.571217 | 0.920217 | 0.763100 |
| noisy_neutral | neutral | 0.793530 | 0.793530 | 59.302208 | 0.773965 | 0.733051 |
| distant_reverb_neutral | neutral | 0.832046 | 0.832046 | 40.596573 | 0.905291 | 0.721494 |
| robotic_monotone | neutral | 0.851837 | 0.851837 | 55.965033 | 0.776099 | 0.744345 |
| whisper_sad | angry | 0.925998 | 0.000727 | 60.782195 | 0.779677 | 0.754031 |
| exaggerated_happy | happy | 0.495681 | 0.495681 | 25.796790 | 0.742115 | 0.651866 |
