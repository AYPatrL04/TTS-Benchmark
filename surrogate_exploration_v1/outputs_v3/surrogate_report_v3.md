# Surrogate Metric Exploration V3

Samples: 26. Target: current `main_metric_0_1`.

## What Was Added

- Target-emotion DSP style fit: compares rate, loudness, f0 range, energy variation, pause rate, and activity against simple happy/sad/angry/neutral acoustic profiles.
- Pause/envelope/spectral features: pause rate, active ratio, dynamic range, envelope jitter, rolloff, bandwidth, low/mid/high band ratios.
- More conservative validation: LOOCV, leave-one-dataset-out, and nested subset LOOCV.

## Top Candidates

| candidate | tier | Pearson | 90% CI | Spearman | 90% CI | Kendall | pairwise acc. | MAE | top5 | bottom5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ridge_reference_components_loo | high_reference | 1.000000 | [1.000,1.000] | 1.000000 | [1.000,1.000] | 1.000000 | 1.000000 | 0.000341 | 1.000000 | 1.000000 |
| ridge_ser_delivery_loo | medium_neural | 0.938212 | [0.892,0.980] | 0.887179 | [0.773,0.944] | 0.710769 | 0.855385 | 0.035349 | 0.600000 | 0.600000 |
| emotion_component_0_1 | medium_neural | 0.888542 | [0.775,0.990] | 0.877607 | [0.741,0.953] | 0.747692 | 0.873846 | 0.258240 | 0.800000 | 0.600000 |
| lodo_ser_delivery | medium_neural | 0.850464 | [0.701,0.983] | 0.800342 | [0.612,0.924] | 0.643077 | 0.821538 | 0.045766 | 0.600000 | 0.600000 |
| ridge_emotion_dsp_text_loo | low_dsp | 0.634198 | [0.454,0.782] | 0.540513 | [0.242,0.753] | 0.396923 | 0.698462 | 0.095619 | 0.400000 | 0.400000 |
| ridge_delivery_low_dsp_loo | low_dsp | 0.633433 | [0.426,0.800] | 0.529573 | [0.238,0.736] | 0.360000 | 0.680000 | 0.091625 | 0.400000 | 0.400000 |
| nested_subset_low_dsp_loo | low_dsp | 0.585509 | [0.406,0.733] | 0.485812 | [0.196,0.684] | 0.316923 | 0.658462 | 0.106340 | 0.400000 | 0.200000 |
| ridge_new_shape_low_dsp_loo | low_dsp | 0.571034 | [0.304,0.778] | 0.423590 | [0.070,0.692] | 0.310769 | 0.655385 | 0.098758 | 0.200000 | 0.400000 |
| prosody_fit_light | low_dsp | 0.616598 | [0.408,0.788] | 0.415483 | [0.053,0.701] | 0.298618 | 0.647692 | 0.251666 | 0.400000 | 0.400000 |
| voice_presence_fit | low_dsp | 0.449350 | [0.107,0.707] | 0.407863 | [0.025,0.698] | 0.298462 | 0.649231 | 0.269534 | 0.400000 | 0.400000 |
| fixed_main_shape_proxy_v1 | low_dsp | 0.466289 | [0.141,0.712] | 0.388034 | [0.014,0.683] | 0.273846 | 0.636923 | 0.151469 | 0.200000 | 0.600000 |
| delivery_fit_v1 | low_dsp | 0.508937 | [0.186,0.773] | 0.351111 | [-0.031,0.670] | 0.249231 | 0.624615 | 0.207468 | 0.400000 | 0.600000 |
| fixed_delivery_surrogate_v1 | low_dsp | 0.508937 | [0.186,0.773] | 0.351111 | [-0.031,0.670] | 0.249231 | 0.624615 | 0.207468 | 0.400000 | 0.600000 |
| lodo_emotion_dsp_text | low_dsp | 0.437062 | [0.109,0.706] | 0.323631 | [-0.037,0.618] | 0.231669 | 0.609231 | 0.128850 | 0.400000 | 0.400000 |

## Current Readout

- Best very-low-cost option: `rate_fit` with Pearson `0.175872`, Spearman `0.186965`, MAE `0.429883`.
- Best low-DSP option by ranking: `ridge_emotion_dsp_text_loo` with Pearson `0.634198`, Spearman `0.540513`, MAE `0.095619`.
- Best low-DSP leave-dataset-out check: `lodo_emotion_dsp_text` with Pearson `0.437062`, Spearman `0.323631`, MAE `0.128850`.
- Best medium-neural option: `ridge_ser_delivery_loo` with Pearson `0.938212`, Spearman `0.887179`, MAE `0.035349`.
- Enhanced low-DSP waveform feature extraction took `0.800s` for `26` clips in this run.

## Best Low-DSP Subsets

| features | size | Pearson | Spearman | MAE | bottom5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `duration_sec+energy_cv+prosody_fit_light+articulation_risk_inverse` | 4 | 0.696879 | 0.703248 | 0.090043 | 0.600000 |
| `text_ease+energy_cv+prosody_fit_light+prosody_activity_light` | 4 | 0.682949 | 0.694359 | 0.093012 | 0.200000 |
| `text_ease+duration_sec+energy_cv+prosody_fit_light` | 4 | 0.680439 | 0.688205 | 0.093989 | 0.600000 |
| `energy_cv+prosody_fit_light+prosody_activity_light+spectral_balance_fit` | 4 | 0.720656 | 0.680684 | 0.088417 | 0.400000 |
| `energy_cv+prosody_fit_light+target_style_fit_v1+articulation_risk_inverse` | 4 | 0.720161 | 0.680000 | 0.085004 | 0.400000 |
| `energy_cv+prosody_fit_light+emotion_arousal_fit_v1+articulation_risk_inverse` | 4 | 0.720556 | 0.678632 | 0.084587 | 0.400000 |
| `energy_cv+prosody_fit_light+articulation_risk_inverse+delivery_fit_v1` | 4 | 0.722692 | 0.677265 | 0.085040 | 0.400000 |
| `duration_sec+energy_cv+prosody_fit_light+spectral_balance_fit` | 4 | 0.701170 | 0.675897 | 0.092295 | 0.400000 |
| `energy_cv+emotion_arousal_fit_v1+pause_naturalness+delivery_fit_v1` | 4 | 0.736761 | 0.671795 | 0.081518 | 0.600000 |
| `speech_rate_wps+energy_cv+prosody_fit_light+spectral_balance_fit` | 4 | 0.732867 | 0.669060 | 0.087359 | 0.600000 |
| `energy_cv+prosody_fit_light+articulation_risk_inverse` | 3 | 0.714966 | 0.669060 | 0.088025 | 0.400000 |
| `speech_rate_wps+energy_cv+high_freq_ratio+prosody_fit_light` | 4 | 0.695952 | 0.667692 | 0.094252 | 0.400000 |

## Interpretation

- The best cheap combinations remain useful as a local ranking filter, but they are not strong enough to replace the current main metric.
- The new target-emotion DSP features help expose style misses, especially sad/happy/angry delivery mismatches, but they still cannot see lexical intelligibility failures as reliably as ASR.
- Medium-neural SER/SIM-like signals are the most promising surrogate family if the goal is to approach the high correlation Yufan reported for SIM versus WER.
- Pure text and duration features are too weak for this task. They can flag difficult prompts, but they do not know whether the generated audio actually pronounced the text or conveyed emotion.
- Leave-dataset-out results are the caution sign: the sample set is still too small and too Parler-specific for a final surrogate claim.
- Both datasets use the same TTS system and speaker. Leave-dataset-out is therefore a boundary-set shift check, not held-out-system validation.

## Local Outputs

- `outputs_v3/surrogate_candidates_v3.csv`
- `outputs_v3/subset_search_top30_v3.csv`
- `outputs_v3/nested_subset_selection_counts.csv`
- `outputs_v3/surrogate_error_analysis_v3.csv`
- `outputs_v3/resource_estimate.csv`
- `outputs_v3/feature_snapshot.csv`
