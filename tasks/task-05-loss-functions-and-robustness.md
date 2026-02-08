# Task 05: Loss functions (ratio/abs + reg) + robustness expectation

## Purpose
블루프린트의 loss를 구현:
- `L_ratio` (in-band 대비 out-of-band separation)
- `L_abs` (absolute efficiency)
- `L_gray` (u*(1-u))
- `L_tv` (total variation)
그리고 robust design을 위해 σ/τ 샘플링에 대한 기대값 형태로 loss를 계산할 수 있게 한다.

## Inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- Task-03 (generator outputs u, X_ste)
- Task-04 (band averages)
- Task-01 (weights/eps)

## Outputs
- (create) `src/crinv/losses.py`
- (create) `tests/test_losses_smoke.py`

## Constraints
- `ε` 안정화 파라미터는 config/constants에서 제어.
- Robust loss는 `E_{σ,τ}`의 Monte Carlo estimate로 구현(샘플 수 config로).

## Acceptance criteria
- 랜덤 입력에서도 loss가 NaN/inf 없이 계산.
- ratio/abs/reg 각 항목 on/off 가능.
- σ/τ 샘플링 횟수 증가 시 loss estimate가 안정화되는 구조.

## Dependencies
- Task-01, Task-03, Task-04

## Verification
- `pytest -q tests/test_losses_smoke.py`

## Rollback notes
- losses.py 및 테스트 삭제.
