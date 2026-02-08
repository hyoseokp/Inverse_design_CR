# Task 02: Symmetry parameterization (A_raw → S)

## Purpose
블루프린트 규칙대로 **대각선 대칭을 loss가 아니라 파라미터화에서 구조적으로 강제**하는 seed 생성 모듈을 구현한다.

## Inputs
- `bot/plans/inverse-design/color-router-inverse-design-blueprint-v1.3.md`
- `bot/plans/inverse-design/repository-structure-mapping.md` (generator/seed_param.py 매핑 참고)
- Task-01 outputs:
  - `bot/plans/inverse-design/code/inverse_design/config.py`

## Outputs
- (create) `bot/plans/inverse-design/code/inverse_design/seed.py`
- (create) `bot/plans/inverse-design/code/tests/test_seed_symmetry.py`

## Constraints
- ❌ 금지: symmetry loss, step 후 symmetrize, upper-triangle hack
- optimizer는 오직 `A_raw`만 업데이트 가능한 형태여야 함.
- `S = 0.5 * (sigmoid(A_raw) + sigmoid(A_raw)^T)` 형태를 정확히 준수.

## Acceptance criteria
- `A_raw`가 임의 텐서여도 결과 `S`는 항상 `S == S.T` (tolerance 내).
- backprop 시나리오에서 `A_raw.grad`가 계산 가능(grad None 아님)하고 형태가 보존됨.

## Dependencies
- Task-01

## Verification
- `pytest -q tests/test_seed_symmetry.py`

## Rollback notes
- `seed.py` 및 테스트 파일 삭제로 원복.
