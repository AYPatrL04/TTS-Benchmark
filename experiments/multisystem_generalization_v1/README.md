# Multi-system Generalization V1

This experiment tests Main and surrogate metrics on a balanced shared-text matrix:

- systems and voices: Parler-TTS Mini/Jenna, Bark Small/English speaker 6, and Windows SAPI/Zira;
- six English texts per system;
- three regular texts and three boundary texts (digits/address, technical acronyms, and a tongue twister);
- neutral delivery target for every clip.

The generated WAV files, manifests, component metrics, Main scores, per-clip surrogate scores, and grouped generalization report are included. The principal result is in `analysis/generalization_report.md`.

## Reproduce

Run from the repository root in the environment containing the optional TTS dependencies.

```powershell
conda run -n mlevolve-win python scripts\generate_multisystem_generalization.py `
  --cases experiments\multisystem_generalization_v1\inputs\shared_cases.csv `
  --output-dir experiments\multisystem_generalization_v1\generated `
  --manifest-csv experiments\multisystem_generalization_v1\inputs\multisystem_manifest.csv `
  --systems parler bark sapi `
  --overwrite
```

Run the standard WER, acoustic-sanity, emotion/prosody, report join, and Main scoring scripts against `inputs/multisystem_manifest.csv`, then run:

```powershell
conda run -n mlevolve-win python surrogate_exploration_v1\analyze_multisystem_generalization.py
```

## Human calibration

Create a blinded copy before listening:

```powershell
conda run -n mlevolve-win python scripts\prepare_blind_listening_set.py `
  --manifest experiments\multisystem_generalization_v1\inputs\multisystem_manifest.csv `
  --output-dir experiments\multisystem_generalization_v1\human_evaluation `
  --overwrite
```

Do not open `private_blind_key.csv` until all ratings are complete.
