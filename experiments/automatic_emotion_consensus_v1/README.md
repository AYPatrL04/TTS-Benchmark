# Automatic Emotion Consensus V1

This experiment evaluates 52 local TTS clips without human labels. It combines
three automatic emotion models, Whisper WER, acoustic sanity, controlled
same-text emotion generation, grouped surrogate validation, and runtime cost.

The resulting scalar is useful for automatic intelligibility/emotion screening,
but the boundary audit shows that it does not yet measure complete TTS
naturalness. Do not use aggregate system means as model rankings.

## Key Outputs

- `analysis/per_clip_scores.csv`: all Main and selected surrogate scores
- `analysis/surrogate_candidates.csv`: LOOCV, leave-dataset-out, and
  leave-system-out agreement
- `analysis/metric_costs.csv`: measured or same-machine normalized cost
- `analysis/automatic_metric_report.md`: generated complete comparison
- `controlled_generation/analysis/per_clip_scores.csv`: same-text emotion audit

## Reproduce Analysis

```powershell
conda run -n TTS python scripts\build_automatic_emotion_manifest.py
conda run -n TTS python scripts\evaluate_automatic_emotion_models.py `
  --input experiments\automatic_emotion_consensus_v1\inputs\evaluation_manifest.csv `
  --output-csv experiments\automatic_emotion_consensus_v1\model_outputs\emotion_model_outputs.csv `
  --embeddings-npz experiments\automatic_emotion_consensus_v1\model_outputs\emotion_embeddings.npz `
  --cost-csv experiments\automatic_emotion_consensus_v1\model_outputs\emotion_model_costs.csv
conda run -n TTS python scripts\combine_automatic_emotion_outputs.py
conda run -n TTS python scripts\analyze_automatic_emotion_consensus.py `
  --input experiments\automatic_emotion_consensus_v1\model_outputs\emotion_model_outputs_all_52.csv `
  --output-dir experiments\automatic_emotion_consensus_v1\analysis
```

Install the additional emotion runtime dependencies from
`requirements-emotion-models.txt`. Model weights are downloaded to the normal
user caches and are not stored in this repository.
