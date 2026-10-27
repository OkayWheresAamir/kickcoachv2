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
