# Prompt (Task-04): Spectral processing utilities (banding + G merge)

## Role
You are the Code Executor. Implement exactly the outputs below.

## Read-only inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- `tasks/task-04-spectral-banding-and-merge.md`
- Task-01 config/constants

## Writable outputs (create)
- `src/crinv/spectral.py`
- `tests/test_spectral_merge_and_bands.py`

## Constraints
- Band ranges must be config/constants-controlled.
- 301->30 downsample must be pluggable later; for now provide a deterministic default.

## Verification
```bash
python -m pytest -q tests/test_spectral_merge_and_bands.py
```
