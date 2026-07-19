# BoundaryMetricV1 Composite Main Metric

All component scores and final scores are normalized to 0-1, higher is better.

## Formula

```text
I = 0.80 * (1 - WER) + 0.20 * (1 - CER)
Q = (naturalness_proxy_1_5 - 1) / 4
E = 0.70 * target_emotion_prob + 0.30 * target_emotion_match
P = 1 - abs(prosody_activity - target_prosody) / tolerance
raw = 0.45 * I + 0.15 * Q + 0.30 * E + 0.10 * P
gate = 0.35 + 0.65 * I
main_metric = raw * gate
```

Prosody targets: happy=0.85/tol0.35, angry=0.9/tol0.35, sad=0.6/tol0.4, neutral=0.8/tol0.4

## Aggregate

- input: `experiments\boundary_metric_v1\combined\boundary_main_metrics.csv`
- samples: 18
- mean main metric: 0.773667
- best score: 0.993317
- worst score: 0.482527

## Ranked Samples

| rank | id | target | predicted | score | I | Q | E | P | WER | target_prob | audio |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | function_word_repetition | neutral | neutral | 0.993317 | 1.000000 | 1.000000 | 0.983825 | 0.981692 | 0.00000000 | 0.976893 | `experiments\boundary_metric_v1\generated\parler_boundary\function_word_repetition.wav` |
| 2 | sad_text_happy_voice | happy | happy | 0.966175 | 1.000000 | 1.000000 | 0.955541 | 0.795123 | 0.00000000 | 0.936487 | `experiments\boundary_metric_v1\generated\parler_boundary\sad_text_happy_voice.wav` |
| 3 | digits_address | neutral | neutral | 0.949625 | 1.000000 | 1.000000 | 0.943943 | 0.664420 | 0.00000000 | 0.919919 | `experiments\boundary_metric_v1\generated\parler_boundary\digits_address.wav` |
| 4 | angry_text_neutral_voice | neutral | neutral | 0.908341 | 1.000000 | 1.000000 | 0.829633 | 0.594510 | 0.00000000 | 0.756618 | `experiments\boundary_metric_v1\generated\parler_boundary\angry_text_neutral_voice.wav` |
| 5 | homophones_minimal_pairs | neutral | neutral | 0.879373 | 0.940317 | 1.000000 | 0.824078 | 0.944978 | 0.06666667 | 0.748683 | `experiments\boundary_metric_v1\generated\parler_boundary\homophones_minimal_pairs.wav` |
| 6 | robotic_monotone | neutral | neutral | 0.854820 | 0.921512 | 1.000000 | 0.896286 | 0.672085 | 0.07692308 | 0.851837 | `experiments\boundary_metric_v1\generated\parler_boundary\robotic_monotone.wav` |
| 7 | noisy_neutral | neutral | neutral | 0.851358 | 0.929333 | 1.000000 | 0.855471 | 0.675050 | 0.08333333 | 0.793530 | `experiments\boundary_metric_v1\generated\parler_boundary\noisy_neutral.wav` |
| 8 | exaggerated_happy | happy | happy | 0.850926 | 1.000000 | 1.000000 | 0.646977 | 0.568334 | 0.00000000 | 0.495681 | `experiments\boundary_metric_v1\generated\parler_boundary\exaggerated_happy.wav` |
| 9 | control_neutral_neutral | neutral | neutral | 0.849007 | 0.939409 | 1.000000 | 0.856664 | 0.540818 | 0.07142857 | 0.795235 | `experiments\boundary_metric_v1\generated\parler_boundary\control_neutral_neutral.wav` |
| 10 | distant_reverb_neutral | neutral | neutral | 0.843101 | 0.918627 | 0.998939 | 0.882432 | 0.622318 | 0.08333333 | 0.832046 | `experiments\boundary_metric_v1\generated\parler_boundary\distant_reverb_neutral.wav` |
| 11 | control_angry_angry | angry | angry | 0.835137 | 1.000000 | 1.000000 | 0.545696 | 0.714286 | 0.00000000 | 0.350994 | `experiments\boundary_metric_v1\generated\parler_boundary\control_angry_angry.wav` |
| 12 | control_happy_happy | happy | neutral | 0.678977 | 1.000000 | 1.000000 | 0.072780 | 0.571429 | 0.00000000 | 0.103971 | `experiments\boundary_metric_v1\generated\parler_boundary\control_happy_happy.wav` |
| 13 | whisper_sad | sad | angry | 0.616864 | 1.000000 | 1.000000 | 0.000509 | 0.167115 | 0.00000000 | 0.000727 | `experiments\boundary_metric_v1\generated\parler_boundary\whisper_sad.wav` |
| 14 | neutral_text_angry_voice | angry | neutral | 0.610671 | 0.929333 | 1.000000 | 0.001478 | 0.714286 | 0.08333333 | 0.002112 | `experiments\boundary_metric_v1\generated\parler_boundary\neutral_text_angry_voice.wav` |
| 15 | happy_text_sad_voice | sad | happy | 0.604236 | 1.000000 | 1.000000 | 0.014120 | 0.000000 | 0.00000000 | 0.020172 | `experiments\boundary_metric_v1\generated\parler_boundary\happy_text_sad_voice.wav` |
| 16 | control_sad_sad | sad | happy | 0.600534 | 1.000000 | 1.000000 | 0.001781 | 0.000000 | 0.00000000 | 0.002545 | `experiments\boundary_metric_v1\generated\parler_boundary\control_sad_sad.wav` |
| 17 | technical_acronyms | neutral | neutral | 0.551019 | 0.588462 | 1.000000 | 0.920404 | 0.613153 | 0.38461538 | 0.886291 | `experiments\boundary_metric_v1\generated\parler_boundary\technical_acronyms.wav` |
| 18 | fast_tongue_twister | neutral | neutral | 0.482527 | 0.567742 | 1.000000 | 0.718650 | 0.500000 | 0.50000000 | 0.598071 | `experiments\boundary_metric_v1\generated\parler_boundary\fast_tongue_twister.wav` |
