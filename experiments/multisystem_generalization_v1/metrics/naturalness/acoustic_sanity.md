# Acoustic Sanity Score

- input: `experiments\multisystem_generalization_v1\inputs\multisystem_manifest.csv`
- samples: 18
- mean acoustic sanity score: 0.995527

This heuristic detects gross acoustic failures. It is not naturalness or MOS and must not be interpreted as either.

| rank | id | acoustic_sanity | rms_dbfs | silence_ratio | clipping_ratio | flatness |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | parler__normal_01 | 1.000000 | -17.019070 | 0.087977 | 0.00000000 | 0.019652 |
| 2 | parler__normal_02 | 1.000000 | -17.105674 | 0.087879 | 0.00000000 | 0.008682 |
| 3 | parler__normal_03 | 1.000000 | -19.883612 | 0.115538 | 0.00000000 | 0.018638 |
| 4 | parler__boundary_digits | 1.000000 | -17.992706 | 0.115686 | 0.00000000 | 0.026377 |
| 5 | parler__boundary_acronyms | 1.000000 | -19.153675 | 0.141287 | 0.00000000 | 0.018368 |
| 6 | bark__normal_01 | 1.000000 | -25.816872 | 0.399723 | 0.00000000 | 0.033184 |
| 7 | bark__normal_02 | 1.000000 | -25.023834 | 0.444857 | 0.00000000 | 0.013318 |
| 8 | bark__normal_03 | 1.000000 | -23.500798 | 0.223938 | 0.00000000 | 0.014001 |
| 9 | bark__boundary_digits | 1.000000 | -25.145469 | 0.360568 | 0.00000000 | 0.032467 |
| 10 | bark__boundary_acronyms | 1.000000 | -25.941878 | 0.380425 | 0.00000000 | 0.016157 |
| 11 | sapi__normal_01 | 1.000000 | -19.626172 | 0.298246 | 0.00000000 | 0.199337 |
| 12 | sapi__normal_02 | 1.000000 | -20.811563 | 0.363420 | 0.00000000 | 0.259605 |
| 13 | sapi__normal_03 | 1.000000 | -20.336213 | 0.335260 | 0.00000000 | 0.222832 |
| 14 | sapi__boundary_digits | 1.000000 | -19.478112 | 0.251543 | 0.00000000 | 0.177021 |
| 15 | sapi__boundary_acronyms | 1.000000 | -19.290407 | 0.204138 | 0.00000000 | 0.125029 |
| 16 | sapi__boundary_tongue_twister | 1.000000 | -20.502030 | 0.388889 | 0.00000000 | 0.287476 |
| 17 | parler__boundary_tongue_twister | 0.998833 | -18.333482 | 0.218310 | 0.00005835 | 0.007746 |
| 18 | bark__boundary_tongue_twister | 0.920659 | -26.534591 | 0.542564 | 0.00000000 | 0.003422 |
