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
| angry | 2 | 0.000 | 0.000 | 0.004 | 0.000 | 0.072 | 0.000 | 0.004 | 0.428 | 0.685 |
| happy | 2 | 0.000 | 0.000 | 0.624 | 1.000 | 0.282 | 0.000 | 0.282 | 0.452 | 0.676 |
| neutral | 2 | 0.977 | 1.000 | 0.775 | 1.000 | 0.526 | 1.000 | 0.775 | 0.342 | 0.625 |
| sad | 2 | 1.000 | 1.000 | 0.010 | 0.000 | 0.256 | 0.000 | 0.256 | 0.373 | 0.648 |

## Surrogate Agreement

| candidate | validation | Spearman | Kendall tau-b | pairwise accuracy | MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| low_dsp_ridge | LOOCV | -0.523810 | -0.285714 | 0.357143 | 0.258241 |
| e2v_plus_dsp_ridge | LOOCV | -0.285714 | -0.142857 | 0.428571 | 0.270317 |
| superb_plus_dsp_ridge | LOOCV | 0.380952 | 0.285714 | 0.642857 | 0.202372 |
| vad_plus_dsp_ridge | LOOCV | 0.500000 | 0.428571 | 0.714286 | 0.165387 |
| all_emotion_plus_dsp_ridge | LOOCV | 0.380952 | 0.214286 | 0.607143 | 0.162259 |

## Aggregate Sanity Check

| group | n | mean I | mean E | mean Main |
| --- | ---: | ---: | ---: | ---: |
| non-boundary | 8 | 0.857 | 0.329 | 0.525 |

A higher boundary mean is not evidence of better audio. Label composition and the weak acoustic-sanity detector confound this aggregate; it must not be used as a system-quality ranking.

## Warm Cost

| metric | sec/clip | speedup vs Main | ingredients |
| --- | ---: | ---: | --- |
| main_auto_v3 | 0.7713 | 1.0x | Whisper+sanity+emotion2vec+SUPERB+MSP-Dim |
| low_dsp_ridge | 0.1148 | 6.7x | waveform/text DSP |
| e2v_plus_dsp_ridge | 0.1867 | 4.1x | emotion2vec+DSP |
| superb_plus_dsp_ridge | 0.1510 | 5.1x | SUPERB+DSP |
| vad_plus_dsp_ridge | 0.1342 | 5.7x | MSP-Dim+DSP |
| all_emotion_plus_dsp_ridge | 0.2423 | 3.2x | three emotion models+DSP |

## Per-Clip Scores

| sample | system | target | WER | E consensus | disagreement | Main v3 | low-DSP surrogate | best neural surrogate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| controlled_emotion_intensity_v1::neutral_subtle | parler | neutral | 0.071 | 0.825 | 0.438 | 0.898 | 0.491 | 0.605 |
| controlled_emotion_intensity_v1::neutral_obvious | parler | neutral | 0.286 | 0.724 | 0.465 | 0.742 | 0.503 | 0.744 |
| controlled_emotion_intensity_v1::happy_obvious | parler | happy | 0.143 | 0.326 | 0.691 | 0.620 | 0.317 | 0.462 |
| controlled_emotion_intensity_v1::sad_obvious | parler | sad | 0.071 | 0.228 | 0.994 | 0.572 | 0.660 | 0.579 |
| controlled_emotion_intensity_v1::sad_subtle | parler | sad | 0.214 | 0.283 | 0.985 | 0.563 | 0.500 | 0.551 |
| controlled_emotion_intensity_v1::happy_subtle | parler | happy | 0.214 | 0.238 | 0.556 | 0.530 | 0.648 | 0.592 |
| controlled_emotion_intensity_v1::angry_subtle | parler | angry | 0.071 | 0.006 | 0.072 | 0.158 | 0.592 | 0.632 |
| controlled_emotion_intensity_v1::angry_obvious | parler | angry | 0.071 | 0.002 | 0.071 | 0.118 | 0.531 | 0.409 |
