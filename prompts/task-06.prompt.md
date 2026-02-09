# Prompt (Task-06): Inverse optimization loop (multi-start + top-K export)

## Role
You are the Code Executor. Implement exactly the outputs below.

## Read-only inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- `repository-structure-mapping.md`
- `tasks/task-06-inverse-optimization-multistart-topk.md`
- Task-02..05 outputs
- `common_context_prompt.md`

## Writable outputs (create)
- `src/crinv/inverse_opt.py`
- `src/crinv/surrogate_interface.py`
- `src/crinv/artifacts.py`
- `src/crinv/progress_logger.py`
- `tests/test_inverse_opt_dryrun.py`
- `data/candidates/.gitkeep`
- `data/progress/.gitkeep`

## Constraints
- Surrogate is frozen during inverse (no train).
- Must write file-based progress artifacts (JSONL + topk npz snapshot) in a configurable dir.
- Randomness must be controllable by seed.

## Verification
```bash
python -m pytest -q tests/test_inverse_opt_dryrun.py
```
