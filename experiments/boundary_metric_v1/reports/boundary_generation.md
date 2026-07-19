# Parler Emotion TTS Generation

This run uses Parler-TTS with one fixed speaker description style per target emotion.

- model: `parler-tts/parler-tts-mini-v1`
- device: `cuda`
- dtype: `torch.float16`
- samples: 18
- seed: 2026
- temperature: 1.0000

| id | target_emotion | duration_sec | audio |
| --- | --- | ---: | --- |
| control_happy_happy | happy | 3.796 | `experiments\boundary_metric_v1\generated\parler_boundary\control_happy_happy.wav` |
| control_sad_sad | sad | 4.702 | `experiments\boundary_metric_v1\generated\parler_boundary\control_sad_sad.wav` |
| control_angry_angry | angry | 4.017 | `experiments\boundary_metric_v1\generated\parler_boundary\control_angry_angry.wav` |
| control_neutral_neutral | neutral | 4.435 | `experiments\boundary_metric_v1\generated\parler_boundary\control_neutral_neutral.wav` |
| happy_text_sad_voice | sad | 4.098 | `experiments\boundary_metric_v1\generated\parler_boundary\happy_text_sad_voice.wav` |
| sad_text_happy_voice | happy | 2.902 | `experiments\boundary_metric_v1\generated\parler_boundary\sad_text_happy_voice.wav` |
| angry_text_neutral_voice | neutral | 3.913 | `experiments\boundary_metric_v1\generated\parler_boundary\angry_text_neutral_voice.wav` |
| neutral_text_angry_voice | angry | 3.204 | `experiments\boundary_metric_v1\generated\parler_boundary\neutral_text_angry_voice.wav` |
| digits_address | neutral | 5.329 | `experiments\boundary_metric_v1\generated\parler_boundary\digits_address.wav` |
| homophones_minimal_pairs | neutral | 3.657 | `experiments\boundary_metric_v1\generated\parler_boundary\homophones_minimal_pairs.wav` |
| function_word_repetition | neutral | 3.239 | `experiments\boundary_metric_v1\generated\parler_boundary\function_word_repetition.wav` |
| technical_acronyms | neutral | 4.296 | `experiments\boundary_metric_v1\generated\parler_boundary\technical_acronyms.wav` |
| fast_tongue_twister | neutral | 3.704 | `experiments\boundary_metric_v1\generated\parler_boundary\fast_tongue_twister.wav` |
| noisy_neutral | neutral | 3.680 | `experiments\boundary_metric_v1\generated\parler_boundary\noisy_neutral.wav` |
| distant_reverb_neutral | neutral | 4.714 | `experiments\boundary_metric_v1\generated\parler_boundary\distant_reverb_neutral.wav` |
| robotic_monotone | neutral | 3.750 | `experiments\boundary_metric_v1\generated\parler_boundary\robotic_monotone.wav` |
| whisper_sad | sad | 3.692 | `experiments\boundary_metric_v1\generated\parler_boundary\whisper_sad.wav` |
| exaggerated_happy | happy | 3.843 | `experiments\boundary_metric_v1\generated\parler_boundary\exaggerated_happy.wav` |
