# ASR-WER Evaluation

- input: `experiments\automatic_emotion_consensus_v1\controlled_generation\generated_manifest.csv`
- ASR model: `openai/whisper-tiny.en`
- samples: 8
- mean WER: 0.14285714
- mean CER: 0.09420290

WER is computed by transcribing generated audio with ASR, normalizing text, then comparing the ASR transcript against the input text.
This is an automatic intelligibility proxy, not a replacement for small human listening checks.

| id | WER | CER | S/D/I | transcript |
| --- | ---: | ---: | --- | --- |
| neutral_subtle | 0.07142857 | 0.10144928 | 1/0/0 | The project outweigh is ready and we will review the results together this afternoon. |
| sad_obvious | 0.07142857 | 0.05797101 | 1/0/0 | The project, Oplie, is ready, and we will review the results together this afternoon. |
| angry_subtle | 0.07142857 | 0.07246377 | 1/0/0 | The project.t is ready and we will review the results together this afternoon. |
| angry_obvious | 0.07142857 | 0.05797101 | 1/0/0 | The project automate is ready and we will review the results together this afternoon. |
| happy_obvious | 0.14285714 | 0.08695652 | 1/0/1 | The project, Off tui, is ready and we will review the results together this afternoon. |
| happy_subtle | 0.21428571 | 0.13043478 | 2/0/1 | The project all the years ready and we will review the results together this afternoon. |
| sad_subtle | 0.21428571 | 0.10144928 | 1/0/2 | The project's dirt pay is ready and we will review the results together this afternoon. |
| neutral_obvious | 0.28571429 | 0.14492754 | 3/0/1 | The project up DI is ready and we will review the result near this afternoon. |
