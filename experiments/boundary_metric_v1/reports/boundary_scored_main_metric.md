# boundary_metric_v1 Provisional Teacher V2

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
Mean provisional teacher: 0.782891

| rank | id | status | eligible | teacher | I | Q | Q source | E | WER | CER | prosody diagnostic |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | function_word_repetition | valid_provisional | 1 | 0.991014 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.976893 | 0.00000000 | 0.00000000 | 0.650205 |
| 2 | sad_text_happy_voice | valid_provisional | 1 | 0.975301 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.936487 | 0.00000000 | 0.00000000 | 0.747225 |
| 3 | digits_address | valid_provisional | 1 | 0.968857 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.919919 | 0.00000000 | 0.00000000 | 0.736143 |
| 4 | angry_text_neutral_voice | valid_provisional | 1 | 0.905351 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.756618 | 0.00000000 | 0.00000000 | 0.792187 |
| 5 | robotic_monotone | valid_provisional | 1 | 0.895373 | 0.923077 | 1.000000 | acoustic_sanity_fallback | 0.851837 | 0.07692308 | 0.08474576 | 0.744345 |
| 6 | distant_reverb_neutral | valid_provisional | 1 | 0.883759 | 0.916667 | 0.998939 | acoustic_sanity_fallback | 0.832046 | 0.08333333 | 0.07352941 | 0.721494 |
| 7 | control_neutral_neutral | valid_provisional | 1 | 0.876718 | 0.928571 | 1.000000 | acoustic_sanity_fallback | 0.795235 | 0.07142857 | 0.01724138 | 0.766453 |
| 8 | noisy_neutral | valid_provisional | 1 | 0.868780 | 0.916667 | 1.000000 | acoustic_sanity_fallback | 0.793530 | 0.08333333 | 0.02000000 | 0.733051 |
| 9 | homophones_minimal_pairs | valid_provisional | 1 | 0.861525 | 0.933333 | 1.000000 | acoustic_sanity_fallback | 0.748683 | 0.06666667 | 0.03174603 | 0.662124 |
| 10 | exaggerated_happy | valid_provisional | 1 | 0.803876 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.495681 | 0.00000000 | 0.00000000 | 0.651866 |
| 11 | control_angry_angry | valid_provisional | 1 | 0.747609 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.350994 | 0.00000000 | 0.00000000 | 0.788443 |
| 12 | technical_acronyms | valid_provisional | 0 | 0.720737 | 0.615385 | 1.000000 | acoustic_sanity_fallback | 0.886291 | 0.38461538 | 0.51923077 | 0.724991 |
| 13 | control_happy_happy | valid_provisional | 1 | 0.651544 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.103971 | 0.00000000 | 0.00000000 | 0.785590 |
| 14 | happy_text_sad_voice | valid_provisional | 1 | 0.618956 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.020172 | 0.00000000 | 0.00000000 | 0.747598 |
| 15 | control_sad_sad | valid_provisional | 1 | 0.612101 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.002545 | 0.00000000 | 0.00000000 | 0.783611 |
| 16 | whisper_sad | valid_provisional | 1 | 0.611394 | 1.000000 | 1.000000 | acoustic_sanity_fallback | 0.000727 | 0.00000000 | 0.00000000 | 0.754031 |
| 17 | neutral_text_angry_voice | valid_provisional | 1 | 0.561007 | 0.916667 | 1.000000 | acoustic_sanity_fallback | 0.002112 | 0.08333333 | 0.02000000 | 0.771837 |
| 18 | fast_tongue_twister | valid_provisional | 0 | 0.538139 | 0.500000 | 1.000000 | acoustic_sanity_fallback | 0.598071 | 0.50000000 | 0.16129032 | 0.763100 |
