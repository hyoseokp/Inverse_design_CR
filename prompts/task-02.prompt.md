# Prompt (Task-02): Symmetry parameterization (A_raw -> S)

## Role
You are the Code Executor. Implement exactly the outputs below.

## Read-only inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- `repository-structure-mapping.md`
- `tasks/task-02-symmetry-parameterization.md`
- Task-01 outputs: `src/crinv/config.py`
- `common_context_prompt.md`

## Writable outputs (create/update)
- (create) `src/crinv/seed.py`
- (create) `tests/test_seed_symmetry.py`
- (update) `src/crinv/__init__.py` (export helper)

## Constraints
- Symmetry must be structural (no symmetry loss, no post-step symmetrize).
- Must implement: `S = 0.5 * (sigmoid(A_raw) + sigmoid(A_raw)^T)`.
- Must allow backprop: `A_raw.grad` exists and is finite after backward.

## Verification (must pass)
```bash
python -m pytest -q tests/test_seed_symmetry.py
```
