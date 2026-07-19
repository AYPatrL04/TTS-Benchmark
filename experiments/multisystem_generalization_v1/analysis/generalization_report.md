# Multi-system Generalization and Overfitting Audit

This audit uses the same six texts across Parler-TTS Mini, Bark Small, and Windows SAPI Zira. Three texts are regular and three are boundary cases. All targets are neutral so cross-system behavior is not confounded by requested emotion labels.

## Validation controls

- Learned surrogates use fixed features and strong ridge regularization (`alpha=10`).
- Standardization and fitting are repeated inside every outer fold.
- LOSO holds out an entire TTS system; LOTO holds out a text; LOBC trains on one boundary condition and tests on the other.
- SIM-like references are rebuilt from training-fold audio only.
- `ser_target_prob` is reported as teacher replication, not independent validation.

## Overall agreement

| candidate | validation | Spearman | pairwise accuracy | MAE |
| --- | --- | ---: | ---: | ---: |
| `fixed_low_dsp` | none | -0.512900 | 0.313725 | 0.214260 |
| `ser_target_prob` | none | 0.884417 | 0.882353 | 0.184045 |
| `loso_low_dsp_ridge` | LOSO | 0.079463 | 0.535948 | 0.161561 |
| `loso_neural_dsp_ridge` | LOSO | 0.616099 | 0.699346 | 0.130764 |
| `loto_low_dsp_ridge` | LOTO | 0.217863 | 0.565359 | 0.116569 |
| `loto_neural_dsp_ridge` | LOTO | 0.616099 | 0.712418 | 0.082565 |
| `lobc_low_dsp_ridge` | LOBC | 0.481156 | 0.676471 | 0.096436 |
| `lobc_neural_dsp_ridge` | LOBC | 0.667699 | 0.732026 | 0.082202 |

## Main metric by system

| system | n | Main | I | E (SER teacher) | sanity | LOSO low-DSP MAE | LOSO neural+DSP MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bark | 6 | 0.933618 | 0.915530 | 0.962042 | 0.986776 | 0.250093 | 0.218071 |
| parler | 6 | 0.754115 | 0.886364 | 0.546295 | 0.999806 | 0.144690 | 0.119135 |
| sapi | 6 | 0.794709 | 0.976190 | 0.509523 | 1.000000 | 0.089899 | 0.055086 |

## Regular versus boundary

| boundary | n | Main | I | E | sanity | fixed low-DSP | LOSO neural+DSP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no | 9 | 0.858183 | 0.977778 | 0.670249 | 1.000000 | 0.797645 | 0.825198 |
| yes | 9 | 0.796778 | 0.874279 | 0.674991 | 0.991055 | 0.726411 | 0.721572 |

## Cost observed in this analysis

- low-DSP extraction: `0.825` s total, `0.046` s/clip
- neural SER/embedding extraction: `6.006` s total, `0.334` s/clip on `cuda`
- Main metric additionally requires Whisper ASR; generation time is excluded from metric cost.

## Interpretation

A large drop from text-level validation to LOSO indicates source overfitting. A neural surrogate can agree with this teacher by copying its SER component, but that does not establish human perceptual validity. Cross-system conclusions must therefore use LOSO results and retain I/E/sanity as separate dimensions.

The automatic teacher cannot determine whether Bark's highly rated output is actually more natural than Zira's synthetic timbre, or how objectionable Parler's acronym errors are to listeners. Use the blind rating template in `human_evaluation/ratings_template.csv` before calibrating the Main scalar or claiming a final surrogate.
