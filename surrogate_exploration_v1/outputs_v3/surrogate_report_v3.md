# Surrogate Metric Exploration V3

Samples: 26. Target: current `main_metric_0_1`.

## What Was Added

- Target-emotion DSP style fit: compares rate, loudness, f0 range, energy variation, pause rate, and activity against simple happy/sad/angry/neutral acoustic profiles.
- Pause/envelope/spectral features: pause rate, active ratio, dynamic range, envelope jitter, rolloff, bandwidth, low/mid/high band ratios.
- More conservative validation: LOOCV, leave-one-dataset-out, and nested subset LOOCV.

## Top Candidates

| candidate | tier | Pearson | 90% CI | Spearman | 90% CI | MAE | RMSE | top3 | top5 | bottom5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ridge_reference_components_loo | high_reference | 0.999020 | [0.999,1.000] | 0.994530 | [0.971,1.000] | 0.004836 | 0.006862 | 1.000000 | 1.000000 | 1.000000 |
| ridge_ser_delivery_loo | medium_neural | 0.923714 | [0.879,0.970] | 0.861880 | [0.718,0.932] | 0.042460 | 0.059714 | 0.333333 | 0.600000 | 0.400000 |
| emotion_component_0_1 | medium_neural | 0.748730 | [0.531,0.970] | 0.737436 | [0.489,0.901] | 0.253532 | 0.352813 | 0.666667 | 0.800000 | 0.600000 |
| ridge_delivery_low_dsp_loo | low_dsp | 0.757884 | [0.577,0.878] | 0.662906 | [0.364,0.850] | 0.082021 | 0.103163 | 0.333333 | 0.200000 | 1.000000 |
| ridge_emotion_dsp_text_loo | low_dsp | 0.708831 | [0.524,0.841] | 0.626090 | [0.327,0.811] | 0.085820 | 0.110147 | 0.333333 | 0.200000 | 0.800000 |
| prosody_fit_light | low_dsp | 0.650867 | [0.476,0.815] | 0.579552 | [0.265,0.774] | 0.243254 | 0.294761 | 0.666667 | 0.400000 | 0.600000 |
| ridge_new_shape_low_dsp_loo | low_dsp | 0.722131 | [0.481,0.879] | 0.568547 | [0.226,0.793] | 0.084846 | 0.112394 | 0.333333 | 0.200000 | 0.800000 |
| lodo_ser_delivery | medium_neural | 0.657633 | [0.366,0.960] | 0.563761 | [0.190,0.869] | 0.069958 | 0.131184 | 0.000000 | 0.600000 | 0.600000 |
| voice_presence_fit | low_dsp | 0.605033 | [0.312,0.800] | 0.563077 | [0.226,0.793] | 0.251883 | 0.281648 | 0.333333 | 0.400000 | 0.800000 |
| nested_subset_low_dsp_loo | low_dsp | 0.629271 | [0.344,0.830] | 0.560342 | [0.232,0.800] | 0.094339 | 0.127459 | 0.333333 | 0.600000 | 0.800000 |
| delivery_fit_v1 | low_dsp | 0.647887 | [0.383,0.841] | 0.540513 | [0.202,0.764] | 0.176409 | 0.194645 | 0.333333 | 0.400000 | 0.600000 |
| fixed_delivery_surrogate_v1 | low_dsp | 0.647887 | [0.383,0.841] | 0.540513 | [0.202,0.764] | 0.176409 | 0.194645 | 0.333333 | 0.400000 | 0.600000 |
| fixed_main_shape_proxy_v1 | low_dsp | 0.605724 | [0.313,0.800] | 0.457778 | [0.087,0.712] | 0.129519 | 0.156219 | 0.000000 | 0.200000 | 0.600000 |
| pause_naturalness | low_dsp | 0.408562 | [0.056,0.669] | 0.433846 | [0.086,0.674] | 0.180176 | 0.213064 | 0.000000 | 0.200000 | 0.600000 |

## Current Readout

- Best very-low-cost option: `text_ease` with Pearson `0.367983`, Spearman `0.078380`, MAE `0.185112`.
- Best low-DSP option by ranking: `ridge_delivery_low_dsp_loo` with Pearson `0.757884`, Spearman `0.662906`, MAE `0.082021`.
- Best low-DSP leave-dataset-out check: `lodo_emotion_dsp_text` with Pearson `0.448080`, Spearman `0.288913`, MAE `0.115474`.
- Best medium-neural option: `ridge_ser_delivery_loo` with Pearson `0.923714`, Spearman `0.861880`, MAE `0.042460`.
- Enhanced low-DSP waveform feature extraction took `0.745s` for `26` clips in this run.

## Best Low-DSP Subsets

| features | size | Pearson | Spearman | MAE | bottom5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rate_fit+speech_rate_wps+prosody_fit_light+voice_presence_fit` | 4 | 0.833830 | 0.840000 | 0.068866 | 0.800000 |
| `speech_rate_wps+energy_cv+prosody_fit_light+voice_presence_fit` | 4 | 0.766632 | 0.805128 | 0.076495 | 0.800000 |
| `signal_quality+energy_cv+dynamic_range_fit+voice_presence_fit` | 4 | 0.733331 | 0.797607 | 0.080529 | 0.800000 |
| `energy_cv+prosody_fit_light+voice_presence_fit` | 3 | 0.725982 | 0.794872 | 0.085007 | 0.800000 |
| `energy_cv+zcr+envelope_stability+voice_presence_fit` | 4 | 0.702879 | 0.789402 | 0.085563 | 0.800000 |
| `signal_quality+energy_cv+voice_presence_fit` | 3 | 0.735753 | 0.788034 | 0.080910 | 0.800000 |
| `signal_quality+energy_cv+envelope_stability+voice_presence_fit` | 4 | 0.728704 | 0.788034 | 0.081633 | 0.800000 |
| `signal_quality+energy_cv+spectral_balance_fit+voice_presence_fit` | 4 | 0.722427 | 0.788034 | 0.083845 | 0.800000 |
| `energy_cv+spectral_flatness+prosody_fit_light+voice_presence_fit` | 4 | 0.714168 | 0.786801 | 0.087421 | 0.800000 |
| `signal_quality+energy_cv+voice_presence_fit+articulation_risk_inverse` | 4 | 0.726016 | 0.785983 | 0.085900 | 0.800000 |
| `energy_cv+voice_presence_fit` | 2 | 0.727034 | 0.784615 | 0.080640 | 0.800000 |
| `energy_cv+spectral_balance_fit+voice_presence_fit` | 3 | 0.711503 | 0.783248 | 0.084665 | 0.800000 |

## Interpretation

- The best cheap combinations remain useful as a local ranking filter, but they are not strong enough to replace the current main metric.
- The new target-emotion DSP features help expose style misses, especially sad/happy/angry delivery mismatches, but they still cannot see lexical intelligibility failures as reliably as ASR.
- Medium-neural SER/SIM-like signals are the most promising surrogate family if the goal is to approach the high correlation Yufan reported for SIM versus WER.
- Pure text and duration features are too weak for this task. They can flag difficult prompts, but they do not know whether the generated audio actually pronounced the text or conveyed emotion.
- Leave-dataset-out results are the caution sign: the sample set is still too small and too Parler-specific for a final surrogate claim.

## Local Outputs

- `outputs_v3/surrogate_candidates_v3.csv`
- `outputs_v3/subset_search_top30_v3.csv`
- `outputs_v3/nested_subset_selection_counts.csv`
- `outputs_v3/surrogate_error_analysis_v3.csv`
- `outputs_v3/resource_estimate.csv`
- `outputs_v3/feature_snapshot.csv`
