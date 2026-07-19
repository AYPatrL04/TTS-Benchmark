# Surrogate Metric Exploration

This directory fits low-compute candidates to `provisional_teacher_v2`. The
target is an automatic teacher, not human perceptual truth.

## Data and Validation Scope

- 26 English clips: 8 regular emotion prompts and 18 boundary cases.
- One Parler-TTS configuration and largely one speaker.
- Fold-pure LOOCV, nested feature-selection LOOCV, and leave-dataset-out evaluation.
- Leave-dataset-out separates regular and boundary sets, but does not hold out a
  TTS system or speaker.

Reported agreement includes Pearson, Spearman, Kendall tau-b, pairwise ranking
accuracy, top/bottom-k overlap, MAE, and RMSE. Pearson alone is not used to
select a ranker.

## Candidate Families

`very_low` uses text and duration. `low_dsp` adds rate, silence, energy, robust
pitch/prosody, pause, spectral balance, and voice-presence features without a
neural model. `medium_neural` adds a cached wav2vec2 SER encoder and SIM-like
embedding features. `high_reference` reuses teacher components and is only a
sanity upper bound.

Pitch/style features now prefer semitone dispersion relative to each utterance
over raw Hz dispersion. Hand-set emotion profiles remain exploratory.

## Current Results

| Candidate | Validation | Tier | Spearman | Kendall | Pairwise acc. | MAE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SIM/SER + low-DSP ridge | fold-pure LOOCV | medium neural | 0.928205 | 0.790769 | 0.895385 | 0.032165 |
| SIM/SER + low-DSP ridge | fold-pure leave-dataset-out | medium neural | 0.389219 | 0.331526 | 0.658462 | 0.100571 |
| SER target probability | direct | medium neural | 0.877607 | 0.747692 | 0.873846 | 0.258240 |
| emotion DSP/text ridge | LOOCV | low DSP | 0.540513 | 0.396923 | 0.698462 | 0.095619 |
| nested subset ridge | nested LOOCV | low DSP | 0.485812 | 0.316923 | 0.658462 | 0.106340 |
| raw SIM centroid ridge | LOOCV | medium neural | 0.169231 | 0.120000 | 0.560000 | 0.135550 |

The strongest candidate uses the same SER model family as the teacher. Its high
LOOCV agreement is therefore teacher replication and should not be described as
independent perceptual validation. Neural reference features and ridge fitting
are rebuilt inside every outer fold. The large drop under fold-pure
leave-dataset-out validation shows weak subset transfer. Raw SIM-like embedding
similarity alone is not useful on this sample.

## Cost

Latest local CUDA timing for 26 clips:

| Pipeline | Seconds/clip | Approx. speedup vs full v1 teacher |
| --- | ---: | ---: |
| full teacher pipeline | 1.268069 | 1.0x |
| low-DSP | 0.075700 | 16.8x |
| SIM-like embedding | 0.146524 | 8.7x |
| SIM/SER + low-DSP | 0.171009 | 7.4x |

Use low-DSP as a failure detector. Use SIM/SER + low-DSP only as an experimental
ranker with fallback to the full pipeline. Neither is ready as a reward or final
benchmark metric.

## Reproduce

```powershell
conda run -n mlevolve-win python surrogate_exploration_v1\analyze_surrogates_v3.py
conda run -n mlevolve-win python surrogate_exploration_v1\measure_metric_costs.py
conda run -n mlevolve-win python surrogate_exploration_v1\analyze_sim_like_surrogates.py
```

Curated outputs are under `outputs_v3`. Intermediate subset-search snapshots
and temporary evaluator outputs are ignored by Git.

## Required Next Step

Freeze a human-calibrated main metric, collect multiple TTS systems and
speakers, split by system and text, and reserve one unseen TTS system for final
testing. Refit all candidates whenever the teacher definition changes.
