# SIM-like Neural Surrogate Exploration

Model: `superb/wav2vec2-base-superb-er`. Device: `cuda`. Samples: `26`.

The SIM-like signal here is not a production speaker-SIM metric with a separate reference speaker recording. It is a local neural audio-embedding proxy: wav2vec2 hidden embeddings are compared with target-emotion/style centroids using cosine similarity.

## Agreement With Main Metric

| candidate | Pearson | Spearman | Kendall | pairwise acc. | MAE | top5 | bottom5 | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ridge_sim_ser_plus_low_dsp_loo` | 0.955335 | 0.928205 | 0.790769 | 0.895385 | 0.032165 | 0.800000 | 0.600000 | fold-pure LOOCV ridge |
| `sim_ser_target_prob` | 0.888542 | 0.877607 | 0.747692 | 0.873846 | 0.258240 | 0.800000 | 0.600000 |  |
| `lodo_sim_ser_plus_low_dsp` | 0.564751 | 0.389219 | 0.331526 | 0.658462 | 0.100571 | 0.400000 | 0.600000 | fold-pure leave-one-dataset-out |
| `lodo_sim_plus_low_dsp` | 0.361495 | 0.322739 | 0.238678 | 0.610769 | 0.157661 | 0.400000 | 0.400000 | fold-pure leave-one-dataset-out |
| `ridge_sim_plus_low_dsp_loo` | 0.150595 | 0.306667 | 0.187692 | 0.593846 | 0.141427 | 0.000000 | 0.200000 | fold-pure LOOCV ridge |
| `ridge_sim_centroid_loo` | -0.032892 | 0.132991 | 0.083077 | 0.541538 | 0.148101 | 0.000000 | 0.200000 | fold-pure LOOCV ridge |
| `sim_target_cos_loo` | -0.261845 | -0.158291 | -0.101538 | 0.449231 | 0.205940 | 0.000000 | 0.200000 |  |
| `sim_target_rank_loo` | -0.192955 | -0.192190 | -0.149146 | 0.435385 | 0.353694 | 0.200000 | 0.200000 |  |
| `sim_target_softmax_loo` | -0.243915 | -0.241026 | -0.187692 | 0.406154 | 0.535562 | 0.000000 | 0.200000 |  |
| `sim_target_margin_loo` | -0.357636 | -0.353846 | -0.255385 | 0.372308 | 0.311305 | 0.200000 | 0.000000 |  |

## Cost

| scenario | seconds / 26 clips | seconds / clip | relative to main |
| --- | ---: | ---: | ---: |
| `main_metric_current_pipeline` | 32.969806 | 1.268069 | 1.000000 |
| `low_dsp_base_plus_v3_features` | 1.968207 | 0.075700 | 0.059697 |
| `sim_like_embedding_only` | 3.809624 | 0.146524 | 0.115549 |
| `sim_like_plus_low_dsp` | 4.446232 | 0.171009 | 0.134858 |

## Takeaway

- Raw centroid SIM-like cosine scores do not fit the current main metric on this dataset.
- The useful neural signal is the classifier/logit side of the same wav2vec2 model, especially when combined with SIM-like margin and low-DSP features.
- This is still much cheaper than the current full main metric because it avoids Whisper ASR, but it is more expensive than pure low-DSP features.
- The strongest candidate reuses the same SER model family as the teacher. Its agreement is teacher replication, not independent perceptual validation.
- Leave-dataset-out still holds out only a test-set type; both sets share Parler-TTS and the same speaker, so no cross-system generalization is established.
