# Automatic Emotion Consensus Experiment

No human labels are used. Target emotion means the generation prompt label, so this experiment measures automatic-model agreement, not human emotion validity.

## Main Metric

```text
I = clamp(1 - WER)
E = median(P_emotion2vec(target), P_SUPERB(target), P_MSP-VAD-anchor(target))
S = acoustic_sanity_score
Main_auto_v3 = I^0.55 * E^0.35 * S^0.10
eligible = I >= 0.70 and S >= 0.50
```

The geometric mean is non-compensatory. Model disagreement remains a diagnostic field and high disagreement must not be presented as confident emotion correctness.

## Emotion Separability

| emotion | n | e2v target P | e2v acc | SUPERB target P | SUPERB acc | VAD target P | VAD acc | consensus | arousal | valence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| angry | 6 | 0.000 | 0.000 | 0.146 | 0.333 | 0.187 | 0.167 | 0.125 | 0.464 | 0.488 |
| happy | 7 | 0.432 | 0.429 | 0.642 | 0.857 | 0.458 | 0.429 | 0.453 | 0.550 | 0.714 |
| neutral | 32 | 0.966 | 0.969 | 0.743 | 0.812 | 0.466 | 1.000 | 0.759 | 0.421 | 0.550 |
| sad | 7 | 0.571 | 0.571 | 0.014 | 0.000 | 0.313 | 0.000 | 0.185 | 0.375 | 0.524 |

## Surrogate Agreement

| candidate | validation | Spearman | Kendall tau-b | pairwise accuracy | MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| low_dsp_ridge | LOOCV | 0.122599 | 0.087481 | 0.543741 | 0.200952 |
| low_dsp_ridge | leave_dataset_out | 0.013874 | 0.027914 | 0.513952 | 0.230946 |
| low_dsp_ridge | leave_system_out | -0.096389 | -0.075415 | 0.462293 | 0.235492 |
| e2v_plus_dsp_ridge | LOOCV | 0.377871 | 0.227753 | 0.613876 | 0.154721 |
| e2v_plus_dsp_ridge | leave_dataset_out | 0.387469 | 0.245542 | 0.621795 | 0.176547 |
| e2v_plus_dsp_ridge | leave_system_out | 0.227110 | 0.140803 | 0.570136 | 0.229425 |
| superb_plus_dsp_ridge | LOOCV | 0.727653 | 0.542986 | 0.771493 | 0.126672 |
| superb_plus_dsp_ridge | leave_dataset_out | 0.585418 | 0.417798 | 0.708899 | 0.168994 |
| superb_plus_dsp_ridge | leave_system_out | 0.440417 | 0.308391 | 0.653846 | 0.252152 |
| vad_plus_dsp_ridge | LOOCV | 0.482497 | 0.348056 | 0.673831 | 0.136879 |
| vad_plus_dsp_ridge | leave_dataset_out | 0.420338 | 0.305776 | 0.652715 | 0.201689 |
| vad_plus_dsp_ridge | leave_system_out | -0.286519 | -0.202112 | 0.398944 | 0.254138 |
| all_emotion_plus_dsp_ridge | LOOCV | 0.866462 | 0.698377 | 0.848793 | 0.077509 |
| all_emotion_plus_dsp_ridge | leave_dataset_out | 0.868327 | 0.708412 | 0.854072 | 0.086136 |
| all_emotion_plus_dsp_ridge | leave_system_out | 0.457953 | 0.315234 | 0.657617 | 0.219962 |

## Aggregate Sanity Check

| group | n | mean I | mean E | mean Main |
| --- | ---: | ---: | ---: | ---: |
| non-boundary | 29 | 0.944 | 0.491 | 0.684 |
| boundary | 23 | 0.895 | 0.664 | 0.765 |

A higher boundary mean is not evidence of better audio. Label composition and the weak acoustic-sanity detector confound this aggregate; it must not be used as a system-quality ranking.

## Warm Cost

| metric | sec/clip | speedup vs Main | ingredients |
| --- | ---: | ---: | --- |
| main_auto_v3 | 0.7169 | 1.0x | Whisper+sanity+emotion2vec+SUPERB+MSP-Dim |
| low_dsp_ridge | 0.0550 | 13.0x | waveform/text DSP |
| e2v_plus_dsp_ridge | 0.0922 | 7.8x | emotion2vec+DSP |
| superb_plus_dsp_ridge | 0.0748 | 9.6x | SUPERB+DSP |
| vad_plus_dsp_ridge | 0.0710 | 10.1x | MSP-Dim+DSP |
| all_emotion_plus_dsp_ridge | 0.1281 | 5.6x | three emotion models+DSP |

## Per-Clip Scores

| sample | system | target | WER | E consensus | disagreement | Main v3 | low-DSP surrogate | best neural surrogate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| multisystem_generalization_v1::bark__boundary_acronyms | bark | neutral | 0.000 | 0.998 | 0.618 | 0.999 | 0.644 | 0.781 |
| parler_emotion_v1::neutral_01 | parler | neutral | 0.000 | 0.989 | 0.506 | 0.996 | 0.743 | 0.911 |
| boundary_metric_v1::function_word_repetition | parler | neutral | 0.000 | 0.977 | 0.520 | 0.992 | 0.781 | 0.939 |
| parler_emotion_v1::happy_02 | parler | happy | 0.000 | 0.961 | 0.123 | 0.986 | 0.627 | 1.000 |
| multisystem_generalization_v1::parler__boundary_digits | parler | neutral | 0.000 | 0.937 | 0.500 | 0.977 | 0.775 | 0.889 |
| boundary_metric_v1::digits_address | parler | neutral | 0.000 | 0.920 | 0.499 | 0.971 | 0.738 | 0.884 |
| parler_emotion_v1::neutral_02 | parler | neutral | 0.083 | 0.979 | 0.517 | 0.946 | 0.676 | 0.888 |
| multisystem_generalization_v1::bark__normal_03 | bark | neutral | 0.000 | 0.853 | 0.594 | 0.946 | 0.717 | 0.876 |
| multisystem_generalization_v1::bark__boundary_digits | bark | neutral | 0.125 | 0.984 | 0.605 | 0.924 | 0.780 | 0.926 |
| multisystem_generalization_v1::sapi__normal_02 | sapi | neutral | 0.000 | 0.769 | 0.515 | 0.912 | 0.728 | 0.905 |
| boundary_metric_v1::angry_text_neutral_voice | parler | neutral | 0.000 | 0.757 | 0.621 | 0.907 | 0.633 | 0.787 |
| boundary_metric_v1::robotic_monotone | parler | neutral | 0.077 | 0.852 | 0.495 | 0.905 | 0.756 | 0.907 |
| parler_emotion_v1::happy_01 | parler | happy | 0.000 | 0.745 | 0.308 | 0.902 | 0.738 | 0.955 |
| multisystem_generalization_v1::parler__normal_02 | parler | neutral | 0.000 | 0.740 | 0.489 | 0.900 | 0.733 | 0.858 |
| controlled_emotion_intensity_v1::neutral_subtle | parler | neutral | 0.071 | 0.825 | 0.438 | 0.898 | 0.656 | 0.856 |
| boundary_metric_v1::distant_reverb_neutral | parler | neutral | 0.083 | 0.832 | 0.484 | 0.894 | 0.638 | 0.891 |
| multisystem_generalization_v1::bark__boundary_tongue_twister | bark | neutral | 0.182 | 0.995 | 0.627 | 0.886 | 1.000 | 1.000 |
| boundary_metric_v1::control_neutral_neutral | parler | neutral | 0.071 | 0.795 | 0.522 | 0.886 | 0.709 | 0.875 |
| multisystem_generalization_v1::parler__normal_03 | parler | neutral | 0.000 | 0.695 | 0.530 | 0.880 | 0.712 | 0.838 |
| boundary_metric_v1::noisy_neutral | parler | neutral | 0.083 | 0.794 | 0.528 | 0.879 | 0.726 | 0.853 |
| boundary_metric_v1::homophones_minimal_pairs | parler | neutral | 0.067 | 0.749 | 0.514 | 0.870 | 0.732 | 0.874 |
| boundary_metric_v1::exaggerated_happy | parler | happy | 0.000 | 0.665 | 0.504 | 0.867 | 0.618 | 0.824 |
| multisystem_generalization_v1::bark__normal_01 | bark | neutral | 0.200 | 0.944 | 0.523 | 0.867 | 0.785 | 1.000 |
| multisystem_generalization_v1::sapi__boundary_acronyms | sapi | neutral | 0.143 | 0.816 | 0.580 | 0.856 | 0.874 | 0.789 |
| multisystem_generalization_v1::sapi__boundary_digits | sapi | neutral | 0.000 | 0.587 | 0.584 | 0.830 | 0.784 | 0.795 |
| multisystem_generalization_v1::bark__normal_02 | bark | neutral | 0.000 | 0.512 | 0.997 | 0.791 | 0.661 | 0.663 |
| multisystem_generalization_v1::parler__normal_01 | parler | neutral | 0.000 | 0.490 | 0.559 | 0.779 | 0.645 | 0.733 |
| multisystem_generalization_v1::sapi__normal_01 | sapi | neutral | 0.000 | 0.484 | 0.737 | 0.776 | 0.780 | 0.736 |
| parler_emotion_v1::angry_01 | parler | angry | 0.000 | 0.431 | 0.481 | 0.745 | 0.693 | 0.565 |
| multisystem_generalization_v1::sapi__normal_03 | sapi | neutral | 0.000 | 0.428 | 0.654 | 0.743 | 0.827 | 0.780 |
| controlled_emotion_intensity_v1::neutral_obvious | parler | neutral | 0.286 | 0.724 | 0.465 | 0.742 | 0.648 | 0.861 |
| boundary_metric_v1::technical_acronyms | parler | neutral | 0.385 | 0.886 | 0.509 | 0.734 | 0.765 | 0.817 |
| multisystem_generalization_v1::sapi__boundary_tongue_twister | sapi | neutral | 0.000 | 0.401 | 0.724 | 0.727 | 0.831 | 0.766 |
| boundary_metric_v1::whisper_sad | parler | sad | 0.000 | 0.389 | 0.999 | 0.719 | 0.838 | 0.589 |
| multisystem_generalization_v1::parler__boundary_tongue_twister | parler | neutral | 0.182 | 0.478 | 0.851 | 0.692 | 0.628 | 0.682 |
| parler_emotion_v1::sad_01 | parler | sad | 0.071 | 0.321 | 0.999 | 0.645 | 0.784 | 0.550 |
| controlled_emotion_intensity_v1::happy_obvious | parler | happy | 0.143 | 0.326 | 0.691 | 0.620 | 0.674 | 0.596 |
| boundary_metric_v1::control_angry_angry | parler | angry | 0.000 | 0.225 | 0.351 | 0.593 | 0.617 | 0.441 |
| controlled_emotion_intensity_v1::sad_obvious | parler | sad | 0.071 | 0.228 | 0.994 | 0.572 | 0.691 | 0.559 |
| boundary_metric_v1::fast_tongue_twister | parler | neutral | 0.500 | 0.598 | 0.527 | 0.571 | 0.775 | 0.824 |
| controlled_emotion_intensity_v1::sad_subtle | parler | sad | 0.214 | 0.283 | 0.985 | 0.563 | 0.615 | 0.571 |
| multisystem_generalization_v1::parler__boundary_acronyms | parler | neutral | 0.500 | 0.494 | 0.667 | 0.534 | 0.854 | 0.708 |
| controlled_emotion_intensity_v1::happy_subtle | parler | happy | 0.214 | 0.238 | 0.556 | 0.530 | 0.701 | 0.533 |
| boundary_metric_v1::sad_text_happy_voice | parler | happy | 0.000 | 0.134 | 0.912 | 0.494 | 0.709 | 0.554 |
| boundary_metric_v1::control_happy_happy | parler | happy | 0.000 | 0.104 | 0.276 | 0.453 | 0.698 | 0.411 |
| parler_emotion_v1::angry_02 | parler | angry | 0.059 | 0.084 | 0.202 | 0.406 | 0.676 | 0.373 |
| parler_emotion_v1::sad_02 | parler | sad | 0.000 | 0.050 | 0.429 | 0.351 | 0.739 | 0.465 |
| boundary_metric_v1::happy_text_sad_voice | parler | sad | 0.000 | 0.020 | 0.138 | 0.255 | 0.648 | 0.296 |
| controlled_emotion_intensity_v1::angry_subtle | parler | angry | 0.071 | 0.006 | 0.072 | 0.158 | 0.682 | 0.250 |
| boundary_metric_v1::control_sad_sad | parler | sad | 0.000 | 0.003 | 0.402 | 0.124 | 0.750 | 0.468 |
| controlled_emotion_intensity_v1::angry_obvious | parler | angry | 0.071 | 0.002 | 0.071 | 0.118 | 0.608 | 0.293 |
| boundary_metric_v1::neutral_text_angry_voice | parler | angry | 0.083 | 0.002 | 0.071 | 0.110 | 0.668 | 0.287 |
