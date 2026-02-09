# Common Context Prompt — Inverse Design (Color Router)

## Project objective
Implement an end-to-end inverse-design pipeline for the **Color Router** structure:
- Symmetric 16×16 real seed parameterization (structural y=x symmetry)
- Rule-based generator → 128×128 binary structure
- Frozen forward surrogate evaluation (128×128 → 2×2×30)
- Inverse optimization (multi-start + robustness)
- Top-K selection → GDS export → Lumerical FDTD validation (Top-M)
- Ranking by FDTD (not surrogate)
- Optional active loop: append FDTD-validated data → fine-tune surrogate → re-run inverse

## Source-of-truth documents (read first)
- Blueprint (algorithm/spec): `color-router-inverse-design-blueprint-v1.3.md`
- Repo structure mapping (module layout): `repository-structure-mapping.md`
- Task specs (execution contract): `tasks/_task-index.md` and `tasks/**`

## Repository structure (must follow)
Follow `repository-structure-mapping.md`.

Key dirs:
- `configs/` — yaml configs (inverse/fdtd/paths)
- `src/crinv/` — python package
- `src/scripts/` — CLI entry points
- `tests/` — unit/smoke tests
- `REPORTS/` — integration verify logs/summaries
- `data/` — large artifacts (recommended git-excluded except examples/.gitkeep)
  - `data/progress/` — dashboard-readable logs/snapshots

## Core invariants (never violate)
1. **Seed symmetry is structural, not penalized**
   - MUST implement `S = 0.5 * (sigmoid(A_raw) + sigmoid(A_raw)^T)`.
   - ❌ Do not use symmetry loss, post-step symmetrize, or upper-triangle hacks.
2. **Single rule-based generator** shared by:
   - training data generation
   - inverse optimization
   - FDTD validation
3. **FDTD validation path is fixed:** `seed → generator → GDS → FDTD`.
4. **Forward surrogate is frozen during inverse** (evaluator only).
5. **Final ranking uses FDTD**, not surrogate.

## Forward surrogate references (GitHub owner = hyoseokp)
- Model/params repo folder: `hyoseokp/data_CR/code/CR_recon/`
  - Always read `catalog.md` first for IO/loader rules.
- Pre-train dataset repo: https://github.com/hyoseokp/data_CR
- Dataset refinement/loader reference:
  - https://github.com/hyoseokp/CR_DL_auto/blob/main/code/CR_recon/data/dataset.py

## FDTD references (GitHub owner = hyoseokp)
- Reference notebook (lumapi patterns): `hyoseokp/data_CR/data_gen.ipynb`

## Dashboard data contract (Task-06 ↔ Task-10)
Optimization must write file-based progress artifacts (lightweight):
- `data/progress/metrics.jsonl` (append-only)
- `data/progress/topk_step-<step>.npz` (periodic)
- (optional) `data/progress/previews/*.png`
Dashboard must only **read** these files (loose coupling).

## Approval gates / safety
Anything that affects external systems or OS-level processes must be gated:
- Running real Lumerical FDTD via `lumapi` (Windows env, process kill/retry) → **Approval required**
- Training/fine-tuning forward model (GPU/time cost) → **Approval required**

## Implementation rules
- Keep functions pure where possible; make IO explicit.
- Prefer deterministic outputs (seed-controlled sampling for σ/τ jitter).
- Log every run with a run id (e.g., `run_YYYYMMDD_HHMMSS`).
- Tests:
  - Each task adds minimal smoke/unit tests.
  - Integration verify writes to `REPORTS/`.

## Definition of done (global)
- Tasks 01–11 can run end-to-end on a mock surrogate (dry-run) with:
  - correct shapes
  - no NaN/inf
  - artifacts written to expected locations
  - dashboard showing loss + top-K previews
- Optional Task-12 active loop supports:
  - Top-K packaging → Top-M FDTD validation → dataset append → fine-tune hook → inverse re-run
