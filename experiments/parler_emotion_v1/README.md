# Parler Emotion V1

Balanced emotion-control experiment for the current main metric.

## Setup

- TTS model: `parler-tts/parler-tts-mini-v1`
- Prompt style: fixed speaker name with emotion-specific descriptions
- Target emotions: `happy`, `sad`, `angry`, `neutral`
- Samples: 8 total, 2 per target emotion
- Output sample rate: 44100 Hz

## Files

```text
inputs/parler_emotion_manifest.csv
generated/parler_emotion/*.wav
metrics/intelligibility/asr_wer.csv
metrics/naturalness/naturalness_proxy.csv
metrics/style_emotion/emotion_prosody.csv
combined/parler_emotion_main_metrics.csv
combined/parler_emotion_scored_main_metric.csv
reports/parler_emotion_scored_main_metric.md
```

## Reproduce

Run from the repository root.

```powershell
conda run -n TTS python scripts\generate_with_parler_emotion.py `
  --texts examples\emotion_target_texts.csv `
  --output-dir experiments\parler_emotion_v1\generated\parler_emotion `
  --manifest-csv experiments\parler_emotion_v1\inputs\parler_emotion_manifest.csv `
  --report-md experiments\parler_emotion_v1\reports\parler_emotion_generation.md `
  --max-prompts 8 `
  --overwrite

conda run -n TTS python scripts\evaluate_wer_with_transformers_whisper.py `
  --input experiments\parler_emotion_v1\inputs\parler_emotion_manifest.csv `
  --output-csv experiments\parler_emotion_v1\metrics\intelligibility\asr_wer.csv `
  --output-md experiments\parler_emotion_v1\metrics\intelligibility\asr_wer.md `
  --model openai/whisper-tiny.en `
  --overwrite

conda run -n TTS python scripts\evaluate_acoustic_naturalness_proxy.py `
  --input experiments\parler_emotion_v1\inputs\parler_emotion_manifest.csv `
  --output-csv experiments\parler_emotion_v1\metrics\naturalness\naturalness_proxy.csv `
  --output-md experiments\parler_emotion_v1\metrics\naturalness\naturalness_proxy.md

conda run -n TTS python scripts\evaluate_emotion_prosody.py `
  --input experiments\parler_emotion_v1\inputs\parler_emotion_manifest.csv `
  --output-csv experiments\parler_emotion_v1\metrics\style_emotion\emotion_prosody.csv `
  --output-md experiments\parler_emotion_v1\metrics\style_emotion\emotion_prosody.md

conda run -n TTS python scripts\build_main_metrics_report.py `
  --intelligibility experiments\parler_emotion_v1\metrics\intelligibility\asr_wer.csv `
  --naturalness experiments\parler_emotion_v1\metrics\naturalness\naturalness_proxy.csv `
  --style experiments\parler_emotion_v1\metrics\style_emotion\emotion_prosody.csv `
  --output-csv experiments\parler_emotion_v1\combined\parler_emotion_main_metrics.csv `
  --output-md experiments\parler_emotion_v1\reports\parler_emotion_main_metrics_report.md

conda run -n TTS python scripts\score_emotion_tts_main_metric.py `
  --input experiments\parler_emotion_v1\combined\parler_emotion_main_metrics.csv `
  --output-csv experiments\parler_emotion_v1\combined\parler_emotion_scored_main_metric.csv `
  --output-md experiments\parler_emotion_v1\reports\parler_emotion_scored_main_metric.md `
  --experiment-name ParlerEmotionV1
```

## Scores

| rank | id | target | predicted | score |
| ---: | --- | --- | --- | ---: |
| 1 | neutral_01 | neutral | neutral | 0.962698 |
| 2 | happy_02 | happy | happy | 0.952333 |
| 3 | happy_01 | happy | happy | 0.916491 |
| 4 | neutral_02 | neutral | neutral | 0.891419 |
| 5 | angry_01 | angry | angry | 0.851983 |
| 6 | angry_02 | angry | neutral | 0.642522 |
| 7 | sad_02 | sad | happy | 0.622155 |
| 8 | sad_01 | sad | happy | 0.560777 |
