# Prompt (Task-10): CLI + integration verify

## Role
You are the Code Executor.

## Read-only inputs
- `tasks/task-10-cli-and-integration-verify.md`

## Writable outputs
- (create) `src/crinv/cli.py`
- (create) `src/scripts/run_inverse.py`

## Constraints
- `python -m scripts.run_inverse --help` must work.
- `python -m scripts.run_inverse --dry-run` must be fast by default.

## Verification
```bash
python -m scripts.run_inverse --help
python -m scripts.run_inverse --dry-run
```
