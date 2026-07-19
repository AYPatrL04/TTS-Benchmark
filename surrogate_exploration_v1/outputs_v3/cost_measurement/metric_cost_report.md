# Metric Cost Measurement

Samples: 26

## Summary

| scenario | seconds total / 26 clips | seconds / clip | relative to main |
| --- | ---: | ---: | ---: |
| `main_metric_current_pipeline` | 56.804452 | 2.184787 | 1.000000 |
| `very_low_text_duration` | 0.000546 | 0.000021 | 0.00000961 |
| `low_dsp_base_plus_v3_features` | 2.362031 | 0.090847 | 0.04158179 |
| `fixed_surrogate_formula_only` | 0.000832 | 0.000032 | 0.00001465 |

## Components

| name | status | seconds total | seconds / clip | notes |
| --- | --- | ---: | ---: | --- |
| `very_low_text_duration_x200` | success | 0.109076 | 0.000021 | in-process timing |
| `low_dsp_base_features` | success | 1.558287 | 0.059934 | in-process timing |
| `low_dsp_enhanced_features_v3` | success | 0.803744 | 0.030913 | in-process timing |
| `fixed_surrogate_formula_x1000` | success | 0.820123 | 0.000032 | in-process timing |
| `main_naturalness_proxy` | success | 1.186913 | 0.045651 | subprocess timing; log=main_naturalness_proxy.log |
| `main_emotion_ser_prosody` | success | 32.084681 | 1.234026 | subprocess timing; log=main_emotion_ser_prosody.log |
| `main_whisper_wer` | success | 23.426323 | 0.901012 | subprocess timing; log=main_whisper_wer.log |
| `main_composite_score_only` | success | 0.106535 | 0.004098 | in-process timing |
