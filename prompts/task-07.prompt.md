# Prompt (Task-07): GDS export

## Role
You are the Code Executor.

## Read-only inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- `repository-structure-mapping.md`
- `tasks/task-07-gds-export.md`

## Writable outputs
- (create) `src/crinv/gds_export.py`
- (create) `tests/test_gds_export_smoke.py`
- (create) `data/gds/.gitkeep`

## Constraints
- Filename and cellname: `structure_{id:05d}`.
- Layer 1 datatype 0.
- Polygonization must be deterministic.
- Must work on Windows even when paths contain non-ASCII (write via ASCII temp then copy).

## Verification
```bash
python -m pytest -q tests/test_gds_export_smoke.py
```
