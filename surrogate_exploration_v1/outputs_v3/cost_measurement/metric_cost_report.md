# Metric Cost Measurement

Samples: 26

## Summary

| scenario | seconds total / 26 clips | seconds / clip | relative to main |
| --- | ---: | ---: | ---: |
| `main_metric_current_pipeline` | 32.969806 | 1.268069 | 1.000000 |
| `very_low_text_duration` | 0.000494 | 0.000019 | 0.00001498 |
| `low_dsp_base_plus_v3_features` | 1.968207 | 0.075700 | 0.05969726 |
| `fixed_surrogate_formula_only` | 0.000702 | 0.000027 | 0.00002129 |

## Components

| name | status | seconds total | seconds / clip | notes |
| --- | --- | ---: | ---: | --- |
| `very_low_text_duration_x200` | success | 0.099513 | 0.000019 | in-process timing |
| `low_dsp_base_features` | success | 1.309292 | 0.050357 | in-process timing |
| `low_dsp_enhanced_features_v3` | success | 0.658915 | 0.025343 | in-process timing |
| `fixed_surrogate_formula_x1000` | success | 0.707310 | 0.000027 | in-process timing |
| `main_acoustic_sanity` | success | 1.196672 | 0.046026 | subprocess timing; log=main_acoustic_sanity.log |
| `main_emotion_ser_prosody` | success | 16.132180 | 0.620468 | subprocess timing; log=main_emotion_ser_prosody.log |
| `main_whisper_wer` | success | 15.541929 | 0.597766 | subprocess timing; log=main_whisper_wer.log |
| `main_composite_score_only` | success | 0.099025 | 0.003809 | in-process timing |
