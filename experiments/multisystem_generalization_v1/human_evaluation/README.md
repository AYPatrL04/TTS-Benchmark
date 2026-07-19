# Blind Listening Instructions

Listen to files in `audio/` without opening `private_blind_key.csv`.

Use one row per rater and clip. For additional raters, duplicate the 18 template rows and set a different anonymous `rater_id`.

- `intelligibility_1_5`: 1 = impossible to recover the sentence; 5 = every intended word is clear.
- `naturalness_1_5`: 1 = unusable or severely synthetic; 5 = natural human-like speech without disturbing artifacts.
- `emotion_match_1_5`: 1 = strongly conflicts with the requested emotion; 5 = clearly matches it. The current set requests neutral delivery.
- `overall_acceptability_1_5`: 1 = reject; 5 = fully acceptable for the intended use.
- `heard_transcript_optional`: recommended for boundary clips; type what you heard without looking up model identity.

Use at least three raters for a pilot and five or more for metric calibration. Do not reveal model, voice, Main score, or surrogate score before ratings are complete. Keep `private_blind_key.csv` separate until all ratings are locked.
