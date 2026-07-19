# SIM-like Neural Surrogate Exploration

Model: `superb/wav2vec2-base-superb-er`. Device: `cuda`. Samples: `26`.

The SIM-like signal here is not a production speaker-SIM metric with a separate reference speaker recording. It is a local neural audio-embedding proxy: wav2vec2 hidden embeddings are compared with target-emotion/style centroids using cosine similarity.

## Agreement With Main Metric

| candidate | Pearson | Spearman | MAE | top5 | bottom5 | notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ridge_sim_ser_plus_low_dsp_loo` | 0.943748 | 0.928205 | 0.039488 | 0.800000 | 0.800000 | LOOCV ridge |
| `lodo_sim_ser_plus_low_dsp` | 0.720266 | 0.739272 | 0.072021 | 0.600000 | 0.600000 | leave-one-dataset-out |
| `sim_ser_target_prob` | 0.745731 | 0.737436 | 0.275427 | 0.800000 | 0.600000 |  |
| `ridge_sim_plus_low_dsp_loo` | 0.473116 | 0.402393 | 0.108157 | 0.000000 | 0.600000 | LOOCV ridge |
| `lodo_sim_plus_low_dsp` | 0.344402 | 0.333655 | 0.143883 | 0.600000 | 0.400000 | leave-one-dataset-out |
| `ridge_sim_centroid_loo` | 0.002307 | 0.029060 | 0.148841 | 0.200000 | 0.200000 | LOOCV ridge |
| `sim_target_rank_loo` | -0.187193 | -0.224281 | 0.352311 | 0.200000 | 0.000000 |  |
| `sim_target_cos_loo` | -0.272174 | -0.245812 | 0.212431 | 0.000000 | 0.200000 |  |
| `sim_target_softmax_loo` | -0.319995 | -0.343590 | 0.528895 | 0.000000 | 0.000000 |  |
| `sim_target_margin_loo` | -0.386861 | -0.381197 | 0.308006 | 0.200000 | 0.000000 |  |

## Cost

| scenario | seconds / 26 clips | seconds / clip | relative to main |
| --- | ---: | ---: | ---: |
| `main_metric_current_pipeline` | 56.804452 | 2.184787 | 1.000000 |
| `low_dsp_base_plus_v3_features` | 2.362031 | 0.090847 | 0.041582 |
| `sim_like_embedding_only` | 3.904094 | 0.150157 | 0.068729 |
| `sim_like_plus_low_dsp` | 4.600160 | 0.176929 | 0.080982 |

## Takeaway

- Raw centroid SIM-like cosine scores do not fit the current main metric on this dataset.
- The useful neural signal is the classifier/logit side of the same wav2vec2 model, especially when combined with SIM-like margin and low-DSP features.
- This is still much cheaper than the current full main metric because it avoids Whisper ASR, but it is more expensive than pure low-DSP features.
