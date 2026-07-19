# ParlerEmotionV1 Composite Main Metric

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

- input: `experiments\parler_emotion_v1\combined\parler_emotion_main_metrics.csv`
- samples: 8
- mean main metric: 0.800047
- best score: 0.962698
- worst score: 0.560777

## Ranked Samples

| rank | id | target | predicted | score | I | Q | E | P | WER | target_prob | audio |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | neutral_01 | neutral | neutral | 0.962698 | 1.000000 | 1.000000 | 0.992211 | 0.650345 | 0.00000000 | 0.988873 | `experiments\parler_emotion_v1\generated\parler_emotion\neutral_01.wav` |
| 2 | happy_02 | happy | happy | 0.952333 | 1.000000 | 1.000000 | 0.972949 | 0.604480 | 0.00000000 | 0.961356 | `experiments\parler_emotion_v1\generated\parler_emotion\happy_02.wav` |
| 3 | happy_01 | happy | happy | 0.916491 | 1.000000 | 1.000000 | 0.821741 | 0.699691 | 0.00000000 | 0.745344 | `experiments\parler_emotion_v1\generated\parler_emotion\happy_01.wav` |
| 4 | neutral_02 | neutral | neutral | 0.891419 | 0.930392 | 1.000000 | 0.985462 | 0.693475 | 0.08333333 | 0.979231 | `experiments\parler_emotion_v1\generated\parler_emotion\neutral_02.wav` |
| 5 | angry_01 | angry | angry | 0.851983 | 1.000000 | 1.000000 | 0.601849 | 0.714286 | 0.00000000 | 0.431213 | `experiments\parler_emotion_v1\generated\parler_emotion\angry_01.wav` |
| 6 | angry_02 | angry | neutral | 0.642522 | 0.947307 | 1.000000 | 0.058641 | 0.714286 | 0.05882353 | 0.083773 | `experiments\parler_emotion_v1\generated\parler_emotion\angry_02.wav` |
| 7 | sad_02 | sad | happy | 0.622155 | 1.000000 | 1.000000 | 0.035206 | 0.115932 | 0.00000000 | 0.050295 | `experiments\parler_emotion_v1\generated\parler_emotion\sad_02.wav` |
| 8 | sad_01 | sad | happy | 0.560777 | 0.939286 | 1.000000 | 0.000727 | 0.109200 | 0.07142857 | 0.001039 | `experiments\parler_emotion_v1\generated\parler_emotion\sad_01.wav` |
