# Ai_champ_screen

This repository hosts a reinforcement-learning style coding task. The agent must repair `advantage.py` by implementing `compute_gae` so that `run_checks()` passes.

## Prerequisites
- Python 3.13+
- [`uv`](https://github.com/astral-sh/uv) installed
- Set `ANTHROPIC_API_KEY` to run with a live Anthropic model. When the key is absent the harness uses a deterministic offline simulator tuned for partial success. If `.secrets/anthropic_key` exists, it will be read automatically unless `DISABLE_KEYFILE` is set (to any value).

## Quick start
Install dependencies:

```bash
uv sync
```

## Manual workflow
1) Prove that online mode really works (fails fast if the key cannot be used):

```bash
uv run python3 main.py doctor
```

2) Deterministic grading modes (no hidden rewriting beyond what you request):

- Grade the current `advantage.py` exactly as-is:

```bash
uv run python3 main.py check_current
```

- Reset `advantage.py` to the buggy starter and grade it (repeatable sanity check):

```bash
uv run python3 main.py check_starter
```

3) Measure pass rate over 10 sequential runs (uses the offline simulator when no API key is set). Each trial prints ONLINE or OFFLINE, elapsed seconds, and an offline reason when applicable:

```bash
NUM_RUNS=10 uv run python3 main.py
```

4) Optional: tighten the estimate with more runs:

```bash
NUM_RUNS=30 uv run python3 main.py
```

To force true online runs, set `ALLOW_OFFLINE_AGENT=0`; the harness exits if no API key is available even after checking `.secrets/anthropic_key`.

Each trial logs its planned mode up front (ONLINE or OFFLINE with a reason) and reports the actual result with elapsed time.

### Command cheat-sheet (copy/paste ready)
- Doctor (requires working key; proves a real online call happens):

```bash
uv run python3 main.py doctor
```

- Grade your current `advantage.py` without rewriting it:

```bash
uv run python3 main.py check_current
```

- Reset `advantage.py` to the starter buggy version, then grade (deterministic sanity):

```bash
uv run python3 main.py check_starter
```

- Force online trials to fail fast when no key is found (set your own key first):

```bash
export ANTHROPIC_API_KEY="$(cat .secrets/anthropic_key)"  # or paste your key
export ALLOW_OFFLINE_AGENT=0
NUM_RUNS=3 uv run python3 main.py
```

- Force offline mode even if a key file exists (useful for comparing behavior):

```bash
DISABLE_KEYFILE=1 ALLOW_OFFLINE_AGENT=1 NUM_RUNS=3 uv run python3 main.py
```

### Tuning knobs
- `OFFLINE_SUCCESS_PROB` controls the offline solver’s probability of writing the correct reference implementation (default `0.35`).
- `MAX_STEPS` defaults to 6 tool-using turns; raise it if you want to allow longer tool loops.
- `TEMPERATURE` (default `0.6`) influences the model’s exploration for code generation.
- Set `ALLOW_OFFLINE_AGENT=0` to force online evaluation when `ANTHROPIC_API_KEY` is present.
