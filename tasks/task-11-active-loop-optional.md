# Task 11 (Optional): Active loop hooks (dataset append + surrogate fine-tune)

## Purpose
옵션 단계: FDTD로 검증된 데이터를 데이터셋에 추가하고, forward surrogate를 fine-tune한 뒤 inverse를 재실행하는 active loop 훅을 만든다.

## Inputs
- `bot/plans/inverse-design/color-router-inverse-design-blueprint-v1.3.md`
- Task-08 FDTD validated (X, T_FDTD)
- Forward surrogate training code/path (user provided)

## Outputs
- (create) `bot/plans/inverse-design/code/inverse_design/active_loop.py`
- (create) `bot/plans/inverse-design/code/inverse_design/dataset_io.py`
- (create) `bot/plans/inverse-design/code/tests/test_dataset_io_smoke.py`

## Constraints
- Surrogate fine-tune은 환경/데이터에 의존하므로, 기본은 “인터페이스 + 스텁”으로 제공.
- 외부 GPU/학습 실행은 승인 필요.

## Acceptance criteria
- dataset append/load 스모크 테스트 통과.

## Dependencies
- Task-08

## Verification
- `pytest -q tests/test_dataset_io_smoke.py`

## Rollback notes
- active_loop/dataset_io 삭제.
