# TTS Benchmark

Emotion-aware TTS benchmark prototype for designing an automatic main metric and
using it later as the target for surrogate metric search.

The repository contains:

- a lightweight emotional TTS generation script using Parler-TTS;
- automatic metric scripts for intelligibility, acoustic sanity, emotion, and
  prosody;
- a vector-first provisional teacher and an experimental `0-1` scalar;
- two reproducible experiment outputs with generated audio.

## Repository Layout

```text
TTS-Benchmark/
  docs/
    conda_environment.md
    metric_design.md
  examples/
    emotion_target_texts.csv
  experiments/
    parler_emotion_v1/
    boundary_metric_v1/
  scripts/
    generate_with_parler_emotion.py
    evaluate_wer_with_transformers_whisper.py
    evaluate_acoustic_naturalness_proxy.py
    evaluate_emotion_prosody.py
    build_main_metrics_report.py
    summarize_emotion_stress.py
    score_emotion_tts_main_metric.py
  src/
    tts_metric_eval/
      text_metrics.py
  pyproject.toml
  requirements.txt
  requirements-optional.txt
```

## Main Metric (Provisional Teacher V2)

The primary output is the component vector `(I, Q, E, prosody diagnostics,
sanity flags)`. A normalized scalar is retained only as a provisional target for
experiments:

```text
I = 1 - normalized_WER
E = target_emotion_probability
with learned MOS: provisional_teacher_v2 = 0.55*I + 0.35*E + 0.10*Q
without learned MOS: provisional_teacher_v2 = (0.55*I + 0.35*E) / 0.90
ranking_eligible = I >= 0.70 and acoustic_sanity_score >= 0.50
```

Details are in:

```text
docs/metric_design.md
```

## Included Experiments

### `experiments/parler_emotion_v1`

Small balanced emotion set:

- 8 generated samples
- 4 target emotions: happy, sad, angry, neutral
- includes generated audio, per-metric outputs, and composite score output

Key output:

```text
experiments/parler_emotion_v1/combined/parler_emotion_scored_main_metric.csv
```

### `experiments/boundary_metric_v1`

Boundary-case stress test:

- 18 generated samples
- includes clean controls, lexical/voice emotion conflicts, ASR normalization
  traps, acoustic quality traps, and style traps
- intended to test where the automatic main metric is reliable or weak

Key output:

```text
experiments/boundary_metric_v1/combined/boundary_scored_main_metric.csv
```

## Setup

The scripts were tested with Python 3.10 in a conda environment.

```powershell
conda create -n TTS python=3.10 pip -y
conda activate TTS
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
python -m pip install -e .
```

If Parler-TTS import triggers unrelated TensorFlow/protobuf issues in a mixed ML
environment, the generation script sets safe import environment variables before
loading the model.

## Reproduce Current Experiments

Run commands from the repository root.

### Generate Parler Emotion V1

```powershell
conda run -n TTS python scripts\generate_with_parler_emotion.py `
  --texts examples\emotion_target_texts.csv `
  --output-dir experiments\parler_emotion_v1\generated\parler_emotion `
  --manifest-csv experiments\parler_emotion_v1\inputs\parler_emotion_manifest.csv `
  --report-md experiments\parler_emotion_v1\reports\parler_emotion_generation.md `
  --max-prompts 8 `
  --overwrite
```

### Evaluate Metrics

```powershell
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
```

### Build Composite Score

```powershell
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

The boundary experiment uses the same pipeline with:

```text
experiments/boundary_metric_v1/inputs/emotion_boundary_cases.csv
```

## Future Works

The current scalar is not human-grounded truth. It is a provisional teacher for
pipeline validation and surrogate research. Current surrogate experiments are
under `surrogate_exploration_v1`; they report LOOCV and leave-dataset-out
agreement, Kendall tau, pairwise accuracy, top/bottom-k overlap, MAE, and cost.

Candidate surrogate directions:

- ASR-light features such as duration, speech rate, silence ratio, and ASR
  confidence;
- acoustic features such as loudness, spectral flatness, semitone pitch/energy dynamics,
  mel distance, or multi-resolution STFT distance;
- lightweight speaker or emotion embeddings;
- codec-token likelihood/SIM if the TTS model exposes codec tokens or logits.

Highest-priority work is a human-rated, multi-system calibration set; learned
MOS/defect models; heterogeneous ASR and SER models; and a fully held-out TTS
system test. Any teacher update requires rerunning the surrogate search.
