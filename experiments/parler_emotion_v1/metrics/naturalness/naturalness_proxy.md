# Acoustic Naturalness Proxy

- input: `experiments\parler_emotion_v1\inputs\parler_emotion_manifest.csv`
- samples: 8
- mean naturalness proxy: 5.000000

This is a lightweight no-reference fallback, not a learned MOS model. It is intended to keep the main-metric pipeline runnable when UTMOS/NISQA weights are unavailable.

| rank | id | naturalness_proxy | rms_dbfs | silence_ratio | clipping_ratio | flatness |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | happy_01 | 5.000000 | -18.524971 | 0.179842 | 0.00000000 | 0.026511 |
| 2 | happy_02 | 5.000000 | -17.076775 | 0.192053 | 0.00000000 | 0.015866 |
| 3 | sad_01 | 5.000000 | -19.626733 | 0.144737 | 0.00000000 | 0.015006 |
| 4 | sad_02 | 5.000000 | -16.453610 | 0.155070 | 0.00000000 | 0.015116 |
| 5 | angry_01 | 5.000000 | -23.421701 | 0.240175 | 0.00000000 | 0.023633 |
| 6 | angry_02 | 5.000000 | -17.921603 | 0.228992 | 0.00000000 | 0.019363 |
| 7 | neutral_01 | 5.000000 | -18.995986 | 0.155488 | 0.00000000 | 0.022428 |
| 8 | neutral_02 | 5.000000 | -17.562493 | 0.100257 | 0.00000000 | 0.023295 |
