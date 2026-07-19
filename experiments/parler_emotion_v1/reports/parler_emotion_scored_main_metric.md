# parler_emotion_v1 Provisional Teacher V2

The benchmark is vector-first. The scalar is retained for experiments, not treated as human-grounded truth.

```text
I = 1 - normalized_WER                 # CER is diagnostic
E = target_emotion_prob                # uncalibrated SER; no argmax bonus
Q = learned_MOS when available; acoustic sanity is diagnostic only
teacher_v2 = 0.55*I + 0.35*E + 0.10*Q_mos  # with learned MOS
teacher_v2 = (0.55*I + 0.35*E) / 0.90      # without learned MOS
eligible = I >= 0.70 and acoustic_sanity >= 0.50
```

Coverage: 8/8 (100.0%).
Acoustic-sanity quality fallbacks: 8/8.
Mean provisional teacher: 0.800961

| rank | id | status | eligible | teacher | I | Q | Q source | E | WER | CER | prosody diagnostic |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | neutral_01 | valid_provisional | 1 | 0.995673 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.988873 | 0.00000000 | 0.00000000 | 0.694447 |
| 2 | happy_02 | valid_provisional | 1 | 0.984972 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.961356 | 0.00000000 | 0.00000000 | 0.724269 |
| 3 | neutral_02 | valid_provisional | 1 | 0.940997 | 0.916667 | 1.000000 | acoustic_sanity_fallback | 0.979231 | 0.08333333 | 0.01470588 | 0.697101 |
| 4 | happy_01 | valid_provisional | 1 | 0.900967 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.745344 | 0.00000000 | 0.00000000 | 0.696049 |
| 5 | angry_01 | valid_provisional | 1 | 0.778805 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.431213 | 0.00000000 | 0.00000000 | 0.755492 |
| 6 | sad_02 | valid_provisional | 1 | 0.630670 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.050295 | 0.00000000 | 0.00000000 | 0.768948 |
| 7 | angry_02 | valid_provisional | 1 | 0.607742 | 0.941176 | 1.000000 | acoustic_sanity_fallback | 0.083773 | 0.05882353 | 0.02816901 | 0.751986 |
| 8 | sad_01 | valid_provisional | 1 | 0.567864 | 0.928571 | 1.000000 | acoustic_sanity_fallback | 0.001039 | 0.07142857 | 0.01785714 | 0.767677 |
