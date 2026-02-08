# Task 03: Rule-based generator + STE threshold

## Purpose
블루프린트의 고정 generator 파이프라인을 코드로 구현:
`S(16) → upsample(128) → gaussian blur(σ) → u → threshold(τ) → X(binary)`
그리고 inverse 내부에서는 **Forward는 hard binary**, **Backward는 STE**로 gradient가 `u`를 통해 흐르도록 한다.

## Inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- `repository-structure-mapping.md` (generator/rule_base.py 매핑 참고)
- Task-01 outputs (config/constants)
- Task-02 output (seed symmetry)

## Outputs
- (create) `src/crinv/generator.py`
- (create) `src/crinv/ops.py` (blur/upsample/STE 유틸)
- (create) `tests/test_generator_shapes.py`
- (create) `tests/test_ste_threshold.py`

## Constraints
- Generator는 **학습 데이터 생성/Inverse/FDTD 검증** 모두 동일하게 재사용 가능한 형태여야 함.
- Surrogate 입력은 항상 binary (forward path).
- Robust design을 위해 σ set / τ jitter를 config로부터 샘플링 가능해야 함.

## Acceptance criteria
- 입력 `S: (B,16,16)` → 출력 `X_hard: (B,128,128)` (0/1) 및 `u: (B,128,128)` 생성.
- STE 텐서(`X_ste`)는 forward에서는 `X_hard`와 동일한 값을 갖고, backward에서는 `u`에 대해 grad 전달.
- σ/τ 샘플링 로직이 reproducible(seed) 옵션을 지원.

## Dependencies
- Task-01, Task-02

## Verification
- `pytest -q tests/test_generator_shapes.py tests/test_ste_threshold.py`

## Rollback notes
- generator/ops 및 관련 테스트 삭제.
