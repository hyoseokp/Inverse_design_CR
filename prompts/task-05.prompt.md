# Prompt (Task-05): Loss functions + robustness expectation

## Role
You are the Code Executor. Implement exactly the outputs below.

## Read-only inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- `tasks/task-05-loss-functions-and-robustness.md`
- Task-03 generator outputs (u, X_ste)
- Task-04 band averages
- Task-01 weights/eps config

## Writable outputs (create)
- `src/crinv/losses.py`
- `tests/test_losses_smoke.py`

## Constraints
- epsilon from config
- robust loss is Monte Carlo estimate over sigma/tau samples

## Verification
```bash
python -m pytest -q tests/test_losses_smoke.py
```
