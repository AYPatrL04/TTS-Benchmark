# ASR-WER Evaluation

- input: `experiments\multisystem_generalization_v1\inputs\multisystem_manifest.csv`
- ASR model: `openai/whisper-tiny.en`
- samples: 18
- mean WER: 0.07397186
- mean CER: 0.05110439

WER is computed by transcribing generated audio with ASR, normalizing text, then comparing the ASR transcript against the input text.
This is an automatic intelligibility proxy, not a replacement for small human listening checks.

| id | WER | CER | S/D/I | transcript |
| --- | ---: | ---: | --- | --- |
| parler__normal_01 | 0.00000000 | 0.00000000 | 0/0/0 | Please send the final report before the meeting starts tomorrow. |
| parler__normal_02 | 0.00000000 | 0.00000000 | 0/0/0 | The garden was quiet after the rain and the air felt cool. |
| parler__normal_03 | 0.00000000 | 0.00000000 | 0/0/0 | I left the blue notebook beside the lamp in the study. |
| parler__boundary_digits | 0.00000000 | 0.00000000 | 0/0/0 | The access code is a 17b9 and the address is 405 North Lake Drive. |
| bark__normal_02 | 0.00000000 | 0.00000000 | 0/0/0 | The garden was quiet after the rain and the air felt cool. |
| bark__normal_03 | 0.00000000 | 0.00000000 | 0/0/0 | I left the blue notebook beside the lamp in the study. |
| bark__boundary_acronyms | 0.00000000 | 0.00000000 | 0/0/0 | The GPU API uses HTTPS JSON and UTF-8 in the final SDK release. |
| sapi__normal_01 | 0.00000000 | 0.00000000 | 0/0/0 | Please send the final report before the meeting starts tomorrow. |
| sapi__normal_02 | 0.00000000 | 0.00000000 | 0/0/0 | The garden was quiet after the rain, and the air felt cool. |
| sapi__normal_03 | 0.00000000 | 0.00000000 | 0/0/0 | I left the blue notebook beside the lamp in the study. |
| sapi__boundary_digits | 0.00000000 | 0.00000000 | 0/0/0 | The access code is a 17b9, and the address is 405 North Lake Drive. |
| sapi__boundary_tongue_twister | 0.00000000 | 0.00000000 | 0/0/0 | Red leather Yellow leather Unique New York Repeated twice without rushing |
| bark__boundary_digits | 0.12500000 | 0.00000000 | 1/1/0 | the access code is a 17b9 and the address is 405 Northlake Drive. |
| sapi__boundary_acronyms | 0.14285714 | 0.02040816 | 1/0/1 | The GPU API uses HD TPS, JSON, and UTF-8 in the final SDK release. |
| parler__boundary_tongue_twister | 0.18181818 | 0.06349206 | 1/1/0 | Redlar, yellow leather, unique New York, repeated twice without rushing |
| bark__boundary_tongue_twister | 0.18181818 | 0.07936508 | 2/0/0 | Red leather Yellow leather Unite New York Repeated twice without Russian |
| bark__normal_01 | 0.20000000 | 0.18518519 | 0/2/0 | the final report before the meeting starts tomorrow. |
| parler__boundary_acronyms | 0.50000000 | 0.57142857 | 6/0/1 | The 4-Shivroa uses Arca, Zinn and Yuzhil 8 in the final Cobo Kuiq release |
