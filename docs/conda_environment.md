# Conda Environment

Recommended environment name:

```powershell
TTS
```

Create and install:

```powershell
conda create -n TTS python=3.10 pip -y
conda activate TTS
python -m pip install -r requirements.txt
python -m pip install -e .
```

The current benchmark pipeline requires:

- Python 3.10
- numpy
- scipy
- torch
- transformers
- sentencepiece
- soundfile
- parler-tts

Optional future dependencies are listed in:

```text
requirements-optional.txt
```

## Notes

The generation script sets these environment variables before importing Parler
or Transformers:

```text
TRANSFORMERS_NO_TF=1
USE_TF=0
TOKENIZERS_PARALLELISM=false
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

This avoids unrelated TensorFlow/protobuf import issues in mixed ML
environments.
