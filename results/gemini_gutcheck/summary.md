# Gemini gut-check (Phase 1)

Model `gemini-3.8-flash`, prompt `e4-v1` (eval/e4_prompt.py), temperature 0,
sampled at min(playback fps, 24) frames per second (the API maximum). No labels exist
yet, so nothing here is scored; contact frame = contact_time_s x playback fps, for
eyeballing only.

| clip | leg | contact s (frame) | knee deg | trunk deg | cannot-determine reason | prompt tokens | cache file |
|---|---|---|---|---|---|---|---|
| c10 | right | 2.542 (76) | 150.0 | 5.0 | None | 7110 | 77cf3f1a5d2494fb76c877eb.json |
| c19 | right | 3.333 (100) | 158.0 | 7.0 | None | 9378 | 9564eb4664ddff3e298ae3fb.json |
| c25 | right | 2.92 (88) | 153.0 | -14.0 | None | 10602 | 3af53ef66c35c8daa5e0a48c.json |

Model notes:

- **c10:** The player approaches from the left, plants his left foot beside the ball, and strikes the ball with his right instep at 2.542 s. At impact, the kicking knee is moderately flexed (~150 deg) and the trunk is leaning slightly forward (~5 deg).
- **c19:** Ball contact occurs at 00:03.333. The player strikes the ball with his right instep while maintaining a slight forward trunk lean over the ball.
- **c25:** The player kicks the ball with his right leg. Initial ball contact occurs at approximately 00:02.92 (frame at 00:02.917). At contact, the kicking knee is moderately extended at approximately 153 degrees, and the trunk leans backward relative to the kicking direction at approximately -14 degrees.
