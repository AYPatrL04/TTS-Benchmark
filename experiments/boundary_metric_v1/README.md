# Boundary Metric V1

Boundary-case stress test for the current emotion-aware TTS main metric.

## Setup

- TTS model: `parler-tts/parler-tts-mini-v1`
- Samples: 18 total
- Case types:
  - clean controls
  - lexical/content emotion vs voice-emotion conflicts
  - ASR normalization traps
  - high-speed intelligibility stress
  - acoustic quality traps
  - style quality traps

## Files

```text
inputs/emotion_boundary_cases.csv
inputs/boundary_manifest.csv
generated/parler_boundary/*.wav
metrics/intelligibility/asr_wer.csv
metrics/naturalness/naturalness_proxy.csv
metrics/style_emotion/emotion_prosody.csv
combined/boundary_main_metrics.csv
combined/boundary_scored_main_metric.csv
reports/boundary_scored_main_metric.md
```

## Reproduce

Run from the repository root.

```powershell
conda run -n TTS python scripts\generate_with_parler_emotion.py `
  --texts experiments\boundary_metric_v1\inputs\emotion_boundary_cases.csv `
  --output-dir experiments\boundary_metric_v1\generated\parler_boundary `
  --manifest-csv experiments\boundary_metric_v1\inputs\boundary_manifest.csv `
  --report-md experiments\boundary_metric_v1\reports\boundary_generation.md `
  --max-prompts 18 `
  --overwrite

conda run -n TTS python scripts\evaluate_wer_with_transformers_whisper.py `
  --input experiments\boundary_metric_v1\inputs\boundary_manifest.csv `
  --output-csv experiments\boundary_metric_v1\metrics\intelligibility\asr_wer.csv `
  --output-md experiments\boundary_metric_v1\metrics\intelligibility\asr_wer.md `
  --model openai/whisper-tiny.en `
  --overwrite

conda run -n TTS python scripts\evaluate_acoustic_naturalness_proxy.py `
  --input experiments\boundary_metric_v1\inputs\boundary_manifest.csv `
  --output-csv experiments\boundary_metric_v1\metrics\naturalness\naturalness_proxy.csv `
  --output-md experiments\boundary_metric_v1\metrics\naturalness\naturalness_proxy.md

conda run -n TTS python scripts\evaluate_emotion_prosody.py `
  --input experiments\boundary_metric_v1\inputs\boundary_manifest.csv `
  --output-csv experiments\boundary_metric_v1\metrics\style_emotion\emotion_prosody.csv `
  --output-md experiments\boundary_metric_v1\metrics\style_emotion\emotion_prosody.md

conda run -n TTS python scripts\build_main_metrics_report.py `
  --intelligibility experiments\boundary_metric_v1\metrics\intelligibility\asr_wer.csv `
  --naturalness experiments\boundary_metric_v1\metrics\naturalness\naturalness_proxy.csv `
  --style experiments\boundary_metric_v1\metrics\style_emotion\emotion_prosody.csv `
  --output-csv experiments\boundary_metric_v1\combined\boundary_main_metrics.csv `
  --output-md experiments\boundary_metric_v1\reports\boundary_main_metrics_report.md

conda run -n TTS python scripts\score_emotion_tts_main_metric.py `
  --input experiments\boundary_metric_v1\combined\boundary_main_metrics.csv `
  --output-csv experiments\boundary_metric_v1\combined\boundary_scored_main_metric.csv `
  --output-md experiments\boundary_metric_v1\reports\boundary_scored_main_metric.md `
  --experiment-name BoundaryMetricV1
```

## Scores

| rank | id | case | target | predicted | score |
| ---: | --- | --- | --- | --- | ---: |
| 1 | function_word_repetition | asr_normalization | neutral | neutral | 0.993317 |
| 2 | sad_text_happy_voice | lexical_voice_conflict | happy | happy | 0.966175 |
| 3 | digits_address | asr_normalization | neutral | neutral | 0.949625 |
| 4 | angry_text_neutral_voice | lexical_voice_conflict | neutral | neutral | 0.908341 |
| 5 | homophones_minimal_pairs | asr_normalization | neutral | neutral | 0.879373 |
| 6 | robotic_monotone | acoustic_quality_trap | neutral | neutral | 0.854820 |
| 7 | noisy_neutral | acoustic_quality_trap | neutral | neutral | 0.851358 |
| 8 | exaggerated_happy | style_quality_trap | happy | happy | 0.850926 |
| 9 | control_neutral_neutral | control | neutral | neutral | 0.849007 |
| 10 | distant_reverb_neutral | acoustic_quality_trap | neutral | neutral | 0.843101 |
| 11 | control_angry_angry | control | angry | angry | 0.835137 |
| 12 | control_happy_happy | control | happy | neutral | 0.678977 |
| 13 | whisper_sad | style_quality_trap | sad | angry | 0.616864 |
| 14 | neutral_text_angry_voice | lexical_voice_conflict | angry | neutral | 0.610671 |
| 15 | happy_text_sad_voice | lexical_voice_conflict | sad | happy | 0.604236 |
| 16 | control_sad_sad | control | sad | happy | 0.600534 |
| 17 | technical_acronyms | asr_normalization | neutral | neutral | 0.551019 |
| 18 | fast_tongue_twister | intelligibility_stress | neutral | neutral | 0.482527 |

## Notes

This experiment is useful for finding where the automatic main metric can be
fooled. It should be kept as a negative-control set when fitting cheaper
surrogate metrics later.
