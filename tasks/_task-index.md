# Inverse Design Task Index (Color Router)

Source spec:
- `bot/plans/inverse-design/color-router-inverse-design-blueprint-v1.3.md`
- **Repo structure mapping:** `bot/plans/inverse-design/repository-structure-mapping.md`

## Testing policy
- (B) Minimal tests per task + final integration verify task.

## Work items (ordered)
1. **Task-01** Parse spec → define project scaffold + config schema
2. **Task-02** Implement symmetry-parameterized seed (A_raw → S)
3. **Task-03** Implement rule-based generator (16→128, blur, threshold, STE)
4. **Task-04** Implement spectral banding + RGB/G merge utilities
5. **Task-05** Implement loss functions (ratio/abs + gray + TV) + robustness sampling
6. **Task-06** Implement inverse optimization loop (multi-start, top-K export)
7. **Task-07** Implement GDS export (binary grid → polygons → gds)
8. **Task-08** Implement Lumerical FDTD runner (lumapi, stability rules, 2-phase commit)
9. **Task-09** Post-processing + ranking by FDTD + surrogate-vs-FDTD error report
10. **Task-10** Local dashboard (loss graph + top-K previews during optimization)
11. **Task-11** End-to-end CLI + integration verify (dry-run + minimal e2e on mock)
12. **Task-12 (Optional)** Active loop: dataset append + surrogate fine-tune hooks

## Dependency graph (DAG)
- Task-01 → {02,03,04}
- {02,03,04} → 05 → 06
- 06 → {07,09}
- 07 → 08 → 09
- 09 → 10
- 10 → 11
- 11 → (Optional) 12

## Reference code (FDTD)
- `hyoseokp/data_CR/data_gen.ipynb`
  - 여기서 `hyoseokp`는 효석의 GitHub 계정(소유자) 이름.
  - FDTD를 Python으로 다루는 예제 코드. Task-08(FDTD runner) 구현 시 반드시 참고.

## Reference code (Forward surrogate)
- `hyoseokp/data_CR/code/CR_recon/`
  - 여기서 `hyoseokp`는 효석의 GitHub 계정(소유자) 이름.
  - 128×128 구조 → 2×2×30 forward surrogate 모델/파라미터 저장 폴더
  - 모델 로딩/입출력 정의는 폴더 내 `catalog.md`를 먼저 읽고 따른다.

## Critical path
01 → 03 → 05 → 06 → 07 → 08 → 09 → 10 → 11

## Status
- Task-01..12: PENDING
