# Prompt (Task-01): Project scaffold + config schema

## Role
You are the Code Executor. Implement exactly the outputs below.

## Read-only inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- `repository-structure-mapping.md`
- `tasks/task-01-project-scaffold-and-config.md`
- `tasks/_task-index.md`
- `common_context_prompt.md`

## Writable outputs (create/update)
- `README.md`
- `pyproject.toml`
- `.gitignore`
- `configs/paths.example.yaml`
- `configs/inverse.yaml`
- `configs/fdtd.yaml`
- `src/crinv/__init__.py`
- `src/crinv/config.py`
- `src/crinv/constants.py`
- `tests/test_config_smoke.py`

## Constraints
- Follow `repository-structure-mapping.md` for folder layout (`configs/`, `src/crinv/`, `tests/`).
- Do not change spec semantics from the blueprint.
- Keep configs minimal but expressive enough to represent:
  - seed size(16), struct size(128)
  - sigma set, tau0, delta_tau
  - band ranges (B/G/R)
  - output channels (30 or 301)
  - epsilon, loss weights
  - optimization params (N_start, N_steps, Adam lr, topk, robustness samples)

## Verification (must pass)
```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -c "from crinv.config import InverseDesignConfig; print(InverseDesignConfig())"
```
