# Prompt (Task-11): Active loop hooks (dataset append + surrogate fine-tune)

## Role
You are the Code Executor.

## Read-only inputs
- `tasks/task-11-active-loop-optional.md`
- `color-router-inverse-design-blueprint-v1.3.md`

## Writable outputs
- (create) `src/crinv/active_loop.py`
- (create) `src/crinv/dataset_io.py`
- (create) `tests/test_dataset_io_smoke.py`

## Constraints
- Fine-tune/training is an interface/stub only.
- dataset append/load must be testable locally.

## Verification
```bash
python -m pytest -q tests/test_dataset_io_smoke.py
```
