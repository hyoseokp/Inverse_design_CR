# Task 09: Post-processing + final ranking (FDTD-first)

## Purpose
블루프린트 규칙대로 최종 선택은 surrogate가 아니라 **FDTD 결과** 기준으로 수행한다.
- FDTD spectra(301) 처리(부호/단위 변환)
- training과 동일한 방식으로 301→30 downsample 훅
- band metrics(D_b, O_b, ratios)
- surrogate vs FDTD error report
- Top-M 최종 디자인 선정

## Inputs
- `bot/plans/inverse-design/color-router-inverse-design-blueprint-v1.3.md`
- Task-04 spectral utilities
- Task-06 surrogate candidates exports
- Task-08 FDTD run outputs (spectra)

## Outputs
- (create) `bot/plans/inverse-design/code/inverse_design/postprocess.py`
- (create) `bot/plans/inverse-design/code/inverse_design/ranking.py`
- (create) `bot/plans/inverse-design/code/tests/test_ranking_smoke.py`
- (create) `bot/plans/inverse-design/code/artifacts/final/.gitkeep`

## Constraints
- downsample 함수는 training과 동일해야 하므로, 기본은 인터페이스/플러그인 형태로 두고 교체 가능하게.

## Acceptance criteria
- mock FDTD/surrogate 데이터로 ranking이 수행되고 결과 리포트가 생성됨.

## Dependencies
- Task-04, Task-06, Task-08

## Verification
- `pytest -q tests/test_ranking_smoke.py`

## Rollback notes
- postprocess/ranking 및 artifacts/final 삭제.
