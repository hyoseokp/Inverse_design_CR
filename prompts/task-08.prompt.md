# Prompt (Task-08): FDTD runner + commit protocol

## Role
You are the Code Executor.

## Read-only inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- `repository-structure-mapping.md`
- `tasks/task-08-lumerical-fdtd-runner-stability.md`

## Writable outputs
- (create) `src/crinv/commit_protocol.py`
- (create) `src/crinv/fdtd_scripts.py`
- (create) `src/crinv/fdtd_runner.py`
- (create) `data/fdtd_runs/.gitkeep`
- (create) `tests/test_commit_protocol.py`

## Constraints
- Provide testable runner logic without lumapi via backend interface.
- Implement 2-phase commit marker valid=0/2/1 and test it.

## Verification
```bash
python -m pytest -q tests/test_commit_protocol.py
```
