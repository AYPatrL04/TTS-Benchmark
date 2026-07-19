# Surrogate Metric Exploration

Exploration package for low-compute surrogate metrics fitted against the current
emotional TTS `main_metric_0_1`.

The repository should only commit curated scripts and v3/SIM-like summary
outputs from this directory. Early v1/v2 scratch outputs and intermediate timing
artifacts are ignored by `.gitignore`.

## Data

Input scored datasets:

```text
experiments/parler_emotion_v1/combined/parler_emotion_scored_main_metric.csv
experiments/boundary_metric_v1/combined/boundary_scored_main_metric.csv
```

Total samples:

```text
26
```

Target:

```text
main_metric_0_1
```

The target main metric combines intelligibility, acoustic naturalness, target
emotion match, and target-dependent prosody fit. Because it contains ASR and SER
components, a strong surrogate must either approximate those signals or accept a
clear quality gap.

## Cost Tiers

| tier | approximate cost | examples |
| --- | --- | --- |
| `very_low` | text parsing and manifest duration only | `text_ease`, `rate_fit` |
| `low_dsp` | CPU waveform features, no ASR/SER/neural model | f0, energy, pause, spectral balance |
| `medium_neural` | one neural audio encoder or classifier | SER component, future SIM/speaker embedding |
| `high_reference` | reuses main metric components | ASR/SER/naturalness components; sanity upper bound only |

## Current Best Results

From `outputs_v3/surrogate_candidates_v3.csv`:

| candidate | tier | Pearson | Spearman | MAE | bottom5 |
| --- | --- | ---: | ---: | ---: | ---: |
| `ridge_reference_components_loo` | high reference | 0.999020 | 0.994530 | 0.004836 | 1.000000 |
| `ridge_ser_delivery_loo` | medium neural | 0.923714 | 0.861880 | 0.042460 | 0.400000 |
| `nested_subset_low_dsp_loo` | low DSP | 0.730144 | 0.684786 | 0.087550 | 0.800000 |
| `ridge_delivery_low_dsp_loo` | low DSP | 0.757884 | 0.662906 | 0.082021 | 1.000000 |
| `prosody_fit_light` | low DSP | 0.650867 | 0.579552 | 0.243254 | 0.600000 |
| `text_ease` | very low | 0.367983 | 0.078380 | 0.185112 | 0.400000 |

Interpretation:

- The best practical low-cost candidates are useful as ranking filters, not as
  replacements for the main metric.
- `nested_subset_low_dsp_loo` is the most defensible low-DSP result because
  feature selection happens inside each held-out fold.
- `ridge_delivery_low_dsp_loo` has slightly better MAE and bottom-5 detection,
  but it is a predefined combination rather than a nested selected model.
- `ridge_ser_delivery_loo` reaches the useful target range and is consistent
  with the prior that SIM/SER-like neural audio embeddings are promising
  surrogate families.
- Pure text/duration metrics are too weak for this task.

## Low-DSP Feature Findings

The v3 script added target-emotion and waveform-shape features:

- `target_style_fit_v1`: target emotion profile versus rate, loudness, f0 range,
  energy variation, pause rate, and activity.
- `pause_naturalness`: silence ratio, pause count, active-ratio fit.
- `spectral_balance_fit`: centroid and high-frequency balance.
- `voice_presence_fit`: mid-band energy plus voiced/active ratio.
- `articulation_risk_inverse`: text difficulty plus rate and spectral risk.
- `delivery_fit_v1`: hand-weighted delivery score combining the above.

Best optimistic subset search results:

| features | Pearson | Spearman | MAE | note |
| --- | ---: | ---: | ---: | --- |
| `energy_cv+target_style_fit_v1+pause_naturalness+delivery_fit_v1` | 0.800939 | 0.777778 | 0.075352 | best global subset; optimistic |
| `text_ease+energy_cv+prosody_fit_light+prosody_activity_light` | 0.759180 | 0.772308 | 0.085614 | v2-style baseline remains strong |
| `text_ease+energy_cv+prosody_fit_light+spectral_balance_fit` | 0.775216 | 0.765470 | 0.079997 | adds useful spectral feature |

Nested subset feature frequency:

| feature | selected fraction |
| --- | ---: |
| `energy_cv` | 1.000000 |
| `prosody_fit_light` | 0.923077 |
| `text_ease` | 0.384615 |
| `articulation_risk_inverse` | 0.269231 |
| `spectral_balance_fit` | 0.269231 |
| `target_style_fit_v1` | 0.076923 |
| `delivery_fit_v1` | 0.076923 |

This suggests the most worth-retaining low-cost ingredients are energy
variation, prosody fit, text difficulty/articulation risk, and spectral balance.
The handcrafted target-emotion profile is useful in some subsets, but is not yet
stable enough to be the core surrogate.

## Resource Notes

From `outputs_v3/resource_estimate.csv`:

```text
Enhanced low-DSP waveform feature extraction for 26 clips: 0.804s
Full v3 exploration run, including subset searches: about 49s
```

The extraction itself is cheap. The slower part is research-time model/feature
selection, which would not be needed in normal benchmark execution.

Additional measured cost comparison is stored in
`outputs_v3/cost_measurement/metric_cost_report.md`.

Measured on the current local 26 clips with cached models and CUDA:

| scenario | total seconds / 26 clips | seconds / clip | relative to main |
| --- | ---: | ---: | ---: |
| `main_metric_current_pipeline` | 56.804452 | 2.184787 | 1.000000 |
| `low_dsp_base_plus_v3_features` | 2.362031 | 0.090847 | 0.041582 |
| `very_low_text_duration` | 0.000546 | 0.000021 | 0.000010 |
| `fixed_surrogate_formula_only` | 0.000832 | 0.000032 | 0.000015 |

Main metric component costs:

| component | total seconds / 26 clips | seconds / clip |
| --- | ---: | ---: |
| `main_emotion_ser_prosody` | 32.084681 | 1.234026 |
| `main_whisper_wer` | 23.426323 | 0.901012 |
| `main_naturalness_proxy` | 1.186913 | 0.045651 |
| `main_composite_score_only` | 0.106535 | 0.004098 |

So the practical low-DSP surrogate is about `24x` faster than the current main
metric pipeline, while text/duration-only and formula-only costs are effectively
free compared with neural ASR/SER.

## SIM-like Neural Signal Test

Additional local experiment:

```text
analyze_sim_like_surrogates.py
outputs_v3/sim_like/
```

Model:

```text
superb/wav2vec2-base-superb-er
```

This is a SIM-like proxy rather than a production speaker-SIM metric: the script
extracts wav2vec2 hidden embeddings and compares each sample to target-emotion
centroids with cosine similarity. It also tests whether the same neural model's
target-emotion probability improves the surrogate.

Agreement with current `main_metric_0_1`:

| candidate | Pearson | Spearman | MAE | top5 | bottom5 | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ridge_sim_ser_plus_low_dsp_loo` | 0.943748 | 0.928205 | 0.039488 | 0.800000 | 0.800000 | best LOOCV result |
| `lodo_sim_ser_plus_low_dsp` | 0.720266 | 0.739272 | 0.072021 | 0.600000 | 0.600000 | stricter leave-dataset-out check |
| `sim_ser_target_prob` | 0.745731 | 0.737436 | 0.275427 | 0.800000 | 0.600000 | neural target-emotion probability alone |
| `ridge_sim_plus_low_dsp_loo` | 0.473116 | 0.402393 | 0.108157 | 0.000000 | 0.600000 | raw embedding SIM + low-DSP, no SER probability |
| `ridge_sim_centroid_loo` | 0.002307 | 0.029060 | 0.148841 | 0.200000 | 0.200000 | raw centroid SIM only |

Cost:

| scenario | total seconds / 26 clips | seconds / clip | relative to main | approximate speedup |
| --- | ---: | ---: | ---: | ---: |
| `main_metric_current_pipeline` | 56.804452 | 2.184787 | 1.000000 | 1.0x |
| `low_dsp_base_plus_v3_features` | 2.362031 | 0.090847 | 0.041582 | 24.0x |
| `sim_like_embedding_only` | 3.904094 | 0.150157 | 0.068729 | 14.5x |
| `sim_like_plus_low_dsp` | 4.600160 | 0.176929 | 0.080982 | 12.3x |

SIM-like result interpretation:

- Pure embedding centroid cosine does not match the current main metric on this
  dataset.
- The high score comes from combining neural target-emotion probability,
  SIM-like embedding margin, and low-DSP features.
- This combination is about `12x` faster than the full main metric and fits
  better than low-DSP alone, but it is no longer a purely cheap DSP surrogate.
- Leave-dataset-out remains much lower than LOOCV, so more samples are still
  needed before making a final surrogate claim.

## Recommendation

Use the following hierarchy for further surrogate work:

1. Main candidate for local low-cost filtering:
   `nested_subset_low_dsp_loo` / `ridge_delivery_low_dsp_loo`.
2. Keep as simple baseline:
   `prosody_fit_light`.
3. Treat as the strongest likely surrogate family:
   SIM/SER-like medium-neural audio embeddings, especially if they can be
   computed without full ASR.
4. Do not use alone:
   text-only, duration-only, or signal-quality-only metrics.

Current conclusion: low-DSP features can identify many bad cases cheaply, but
they do not yet fit the main metric well enough for final benchmark selection.
The next high-value step is to add a real SIM/speaker-style embedding or
codec-logit likelihood feature and compare it against the current low-DSP
baseline with the same LOOCV, leave-dataset-out, and top/bottom-k checks.

## Files

```text
analyze_surrogates.py
analyze_surrogates_v2.py
analyze_surrogates_v3.py
analyze_sim_like_surrogates.py
measure_metric_costs.py
outputs/
outputs_v2/
outputs_v3/
```

Commit-ready outputs:

```text
outputs_v3/surrogate_candidates_v3.csv
outputs_v3/surrogate_error_analysis_v3.csv
outputs_v3/resource_estimate.csv
outputs_v3/surrogate_report_v3.md
outputs_v3/sim_like/sim_like_candidates.csv
outputs_v3/sim_like/sim_like_costs.csv
outputs_v3/sim_like/sim_like_per_sample.csv
outputs_v3/sim_like/sim_like_report.md
outputs_v3/cost_measurement/metric_cost_measurements.csv
outputs_v3/cost_measurement/metric_cost_report.md
outputs_v3/cost_measurement/metric_cost_summary.csv
```

Ignored local scratch outputs:

```text
outputs/
outputs_v2/
outputs_v3/cost_measurement/asr_wer.*
outputs_v3/cost_measurement/emotion_prosody.*
outputs_v3/cost_measurement/naturalness_proxy.*
outputs_v3/cost_measurement/*.log
outputs_v3/feature_snapshot.csv
outputs_v3/nested_subset_selection_counts.csv
outputs_v3/subset_search_top30_v3.csv
```
