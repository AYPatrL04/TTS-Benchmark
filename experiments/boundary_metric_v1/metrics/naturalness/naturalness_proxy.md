# Acoustic Naturalness Proxy

- input: `experiments\boundary_metric_v1\inputs\boundary_manifest.csv`
- samples: 18
- mean naturalness proxy: 4.999764

This is a lightweight no-reference fallback, not a learned MOS model. It is intended to keep the main-metric pipeline runnable when UTMOS/NISQA weights are unavailable.

| rank | id | naturalness_proxy | rms_dbfs | silence_ratio | clipping_ratio | flatness |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | control_happy_happy | 5.000000 | -18.679171 | 0.209549 | 0.00000000 | 0.027110 |
| 2 | control_sad_sad | 5.000000 | -19.989544 | 0.190171 | 0.00000000 | 0.015515 |
| 3 | control_angry_angry | 5.000000 | -17.257601 | 0.197995 | 0.00000000 | 0.022256 |
| 4 | control_neutral_neutral | 5.000000 | -17.874875 | 0.179138 | 0.00000000 | 0.013025 |
| 5 | happy_text_sad_voice | 5.000000 | -15.639452 | 0.167076 | 0.00000000 | 0.015626 |
| 6 | sad_text_happy_voice | 5.000000 | -14.983550 | 0.104167 | 0.00000000 | 0.017404 |
| 7 | angry_text_neutral_voice | 5.000000 | -18.495793 | 0.089974 | 0.00000000 | 0.022509 |
| 8 | neutral_text_angry_voice | 5.000000 | -21.108702 | 0.144654 | 0.00000000 | 0.019611 |
| 9 | digits_address | 5.000000 | -17.576236 | 0.139623 | 0.00000000 | 0.021025 |
| 10 | homophones_minimal_pairs | 5.000000 | -16.074335 | 0.137741 | 0.00000000 | 0.013308 |
| 11 | function_word_repetition | 5.000000 | -18.675702 | 0.221184 | 0.00000000 | 0.010670 |
| 12 | technical_acronyms | 5.000000 | -21.146269 | 0.112412 | 0.00000000 | 0.026442 |
| 13 | fast_tongue_twister | 5.000000 | -16.719772 | 0.073370 | 0.00000000 | 0.049039 |
| 14 | noisy_neutral | 5.000000 | -16.843672 | 0.101093 | 0.00000000 | 0.016528 |
| 15 | robotic_monotone | 5.000000 | -23.307325 | 0.147453 | 0.00000000 | 0.019572 |
| 16 | whisper_sad | 5.000000 | -18.844350 | 0.201635 | 0.00000000 | 0.012292 |
| 17 | exaggerated_happy | 5.000000 | -17.949140 | 0.109948 | 0.00000000 | 0.014708 |
| 18 | distant_reverb_neutral | 4.995757 | -17.664939 | 0.185501 | 0.00005304 | 0.017916 |
