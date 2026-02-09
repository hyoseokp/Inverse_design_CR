# Prompt (Task-09): Postprocess + ranking (FDTD-first)

## Role
You are the Code Executor.

## Read-only inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- `tasks/task-09-postprocess-and-ranking.md`
- Task-04 spectral utilities

## Writable outputs
- (create) `src/crinv/postprocess.py`
- (create) `src/crinv/ranking.py`
- (create) `tests/test_ranking_smoke.py`
- (create) `data/final/.gitkeep`

## Constraints
- Ranking uses FDTD first.
- downsample remains pluggable; use deterministic default for now.

## Verification
```bash
python -m pytest -q tests/test_ranking_smoke.py
```
