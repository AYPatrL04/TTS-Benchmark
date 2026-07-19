# MultisystemGeneralizationV1 Provisional Teacher V2

The benchmark is vector-first. The scalar is retained for experiments, not treated as human-grounded truth.

```text
I = 1 - normalized_WER                 # CER is diagnostic
E = target_emotion_prob                # uncalibrated SER; no argmax bonus
Q = learned_MOS when available; acoustic sanity is diagnostic only
teacher_v2 = 0.55*I + 0.35*E + 0.10*Q_mos  # with learned MOS
teacher_v2 = (0.55*I + 0.35*E) / 0.90      # without learned MOS
eligible = I >= 0.70 and acoustic_sanity >= 0.50
```

Coverage: 18/18 (100.0%).
Acoustic-sanity quality fallbacks: 18/18.
Mean provisional teacher: 0.827480

| rank | id | status | eligible | teacher | I | Q | Q source | E | WER | CER | prosody diagnostic |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | bark__boundary_acronyms | valid_provisional | 1 | 0.999286 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.998164 | 0.00000000 | 0.00000000 | 0.867451 |
| 2 | bark__normal_02 | valid_provisional | 1 | 0.999240 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.998047 | 0.00000000 | 0.00000000 | 0.777517 |
| 3 | parler__boundary_digits | valid_provisional | 1 | 0.975307 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.936503 | 0.00000000 | 0.00000000 | 0.660141 |
| 4 | bark__normal_03 | valid_provisional | 1 | 0.942853 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.853051 | 0.00000000 | 0.00000000 | 0.769735 |
| 5 | bark__boundary_digits | valid_provisional | 1 | 0.917403 | 0.875000 | 1.000000 | acoustic_sanity_fallback | 0.984037 | 0.12500000 | 0.00000000 | 0.801795 |
| 6 | sapi__normal_02 | valid_provisional | 1 | 0.910061 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.768729 | 0.00000000 | 0.00000000 | 0.674410 |
| 7 | parler__normal_02 | valid_provisional | 1 | 0.898731 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.739594 | 0.00000000 | 0.00000000 | 0.699519 |
| 8 | bark__boundary_tongue_twister | valid_provisional | 1 | 0.886819 | 0.818182 | 0.920659 | acoustic_sanity_fallback | 0.994678 | 0.18181818 | 0.07936508 | 0.837990 |
| 9 | parler__normal_03 | valid_provisional | 1 | 0.881283 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.694727 | 0.00000000 | 0.00000000 | 0.701526 |
| 10 | bark__normal_01 | valid_provisional | 1 | 0.856106 | 0.800000 | 1.000000 | acoustic_sanity_fallback | 0.944272 | 0.20000000 | 0.18518519 | 0.812187 |
| 11 | sapi__boundary_acronyms | valid_provisional | 1 | 0.841336 | 0.857143 | 1.000000 | acoustic_sanity_fallback | 0.816497 | 0.14285714 | 0.02040816 | 0.651769 |
| 12 | sapi__boundary_digits | valid_provisional | 1 | 0.839338 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.586868 | 0.00000000 | 0.00000000 | 0.619814 |
| 13 | parler__normal_01 | valid_provisional | 1 | 0.776427 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.425097 | 0.00000000 | 0.00000000 | 0.752040 |
| 14 | sapi__normal_03 | valid_provisional | 1 | 0.745635 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.345919 | 0.00000000 | 0.00000000 | 0.645481 |
| 15 | sapi__boundary_tongue_twister | valid_provisional | 1 | 0.718570 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.276323 | 0.00000000 | 0.00000000 | 0.687532 |
| 16 | sapi__normal_01 | valid_provisional | 1 | 0.713312 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.262801 | 0.00000000 | 0.00000000 | 0.677561 |
| 17 | parler__boundary_tongue_twister | valid_provisional | 1 | 0.558023 | 0.818182 | 0.998833 | acoustic_sanity_fallback | 0.149201 | 0.18181818 | 0.06349206 | 0.766458 |
| 18 | parler__boundary_acronyms | valid_provisional | 0 | 0.434918 | 0.500000 | 1.000000 | acoustic_sanity_fallback | 0.332647 | 0.50000000 | 0.57142857 | 0.727245 |
