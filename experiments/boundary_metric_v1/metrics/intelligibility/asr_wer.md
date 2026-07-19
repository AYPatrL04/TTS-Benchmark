# ASR-WER Evaluation

- input: `experiments\boundary_metric_v1\inputs\boundary_manifest.csv`
- ASR model: `openai/whisper-tiny.en`
- samples: 18
- mean WER: 0.07497965
- mean CER: 0.05154354

WER is computed by transcribing generated audio with ASR, normalizing text, then comparing the ASR transcript against the input text.
This is an automatic intelligibility proxy, not a replacement for small human listening checks.

| id | WER | CER | S/D/I | transcript |
| --- | ---: | ---: | --- | --- |
| control_happy_happy | 0.00000000 | 0.00000000 | 0/0/0 | I just got the best news today and I cannot stop smiling. |
| control_sad_sad | 0.00000000 | 0.00000000 | 0/0/0 | I miss the quiet mornings we used to share and the room feels empty now. |
| control_angry_angry | 0.00000000 | 0.00000000 | 0/0/0 | This delay is unacceptable and I need you to fix the problem right now. |
| happy_text_sad_voice | 0.00000000 | 0.00000000 | 0/0/0 | This is wonderful news and everyone should be smiling today. |
| sad_text_happy_voice | 0.00000000 | 0.00000000 | 0/0/0 | The room feels empty today and I really miss you. |
| angry_text_neutral_voice | 0.00000000 | 0.00000000 | 0/0/0 | I told you this was unacceptable and I am extremely frustrated. |
| digits_address | 0.00000000 | 0.00000000 | 0/0/0 | The access code is a 17B9 and the address is 405 North Lake Drive. |
| function_word_repetition | 0.00000000 | 0.00000000 | 0/0/0 | can you can the can or can we can it later if we can? |
| whisper_sad | 0.00000000 | 0.00000000 | 0/0/0 | I'm sorry. I did not know how much that moment meant to you. |
| exaggerated_happy | 0.00000000 | 0.00000000 | 0/0/0 | That was amazing, wonderful, and unbelievably exciting for everyone here. |
| homophones_minimal_pairs | 0.06666667 | 0.03174603 | 1/0/0 | Please write the right word then read the red label and leave the team forward. |
| control_neutral_neutral | 0.07142857 | 0.01724138 | 1/0/0 | The meeting starts at 9.30 and the report is saved in the project folder. |
| robotic_monotone | 0.07692308 | 0.08474576 | 1/0/0 | The system completed the OP and stored the final log in the archive. |
| neutral_text_angry_voice | 0.08333333 | 0.02000000 | 1/0/0 | The package arrived at the front desk at 3.15 in the afternoon. |
| noisy_neutral | 0.08333333 | 0.02000000 | 1/0/0 | The package arrived at the front desk at 3.15 in the afternoon. |
| distant_reverb_neutral | 0.08333333 | 0.07352941 | 1/0/0 | Please open the settings menu, choose account preferences, and confirm the line. |
| technical_acronyms | 0.38461538 | 0.51923077 | 5/0/0 | the pocha loads jump from dropping, then sends twice requests to the charter. |
| fast_tongue_twister | 0.50000000 | 0.16129032 | 4/3/0 | She sells seashells by the seashore then swiftly switches six six groups. |
