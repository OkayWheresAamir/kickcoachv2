# KickCoach v2

> Work in progress. This README is rewritten in Phase 6 with the actual evaluation results.

KickCoach measures a footballer's instep kick from ordinary side-on video. It compares the measurements with cited biomechanics norms and the player's own history, and explains faults with sourced coaching cues. Every measurement is validated against human-labelled ground truth, and a frontier video model is evaluated as a baseline on the same labels.

v2 is an engineering rebuild of [KickCoach](https://github.com/OkayWheresAamir/kickcoach-synapse.ai), which won 3rd place at the Synapse AI Hackathon (DTU, Feb 2026). The audit of the hackathon version is in [docs/V1_AUDIT.md](docs/V1_AUDIT.md).

## Development

```bash
uv sync                 # create .venv and install
uv run pytest           # tests
uv run ruff check .     # lint
```

### Rebuilding the clip data (Phase 1)

Video and frames are never committed; these rebuild them from `data/manifest.csv`.

```bash
uv run python tools/fetch_clips.py      # download sources to data/raw/src/ (YouTube, needs node)
uv run python tools/ingest_clips.py     # trim, extract frames + metadata, decide fps_trusted
uv run python tools/run_pose.py         # MoveNet, RTMPose, MediaPipe -> data/cache/pose/
```

Model weights (gitignored, see `docs/DECISIONS.md` DEC-011):
- MoveNet: the hackathon's `3.tflite`, copied to `models/movenet/v1_3.tflite`.
- RTMPose: rtmlib downloads its ONNX files to `~/.cache/rtmlib` on first use.
- MediaPipe: `curl -L -o models/mediapipe/pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task`

