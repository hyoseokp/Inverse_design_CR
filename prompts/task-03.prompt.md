# Prompt (Task-03): Rule-based generator + STE threshold

## Role
You are the Code Executor. Implement exactly the outputs below.

## Read-only inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- `repository-structure-mapping.md`
- `tasks/task-03-rule-based-generator-ste.md`
- `common_context_prompt.md`

## Writable outputs (create/update)
- (create) `src/crinv/generator.py`
- (create) `src/crinv/ops.py`
- (create) `tests/test_generator_shapes.py`
- (create) `tests/test_ste_threshold.py`

## Constraints
- Generator is the single source of truth for training/inverse/FDTD validation.
- Surrogate input is always binary in forward path.
- In inverse: STE must pass gradient through `u`.
- Must support robust sampling of sigma/tau with reproducibility (seed option).

## Verification (must pass)
```bash
python -m pytest -q tests/test_generator_shapes.py tests/test_ste_threshold.py
```
