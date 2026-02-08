# Task 04: Spectral processing utilities (banding + G merge)

## Purpose
Surrogate/FDTD 결과 스펙트럼에서
- RGGB → G 병합(`T_G = (T_01 + T_10)/2`)
- 30채널 band average(가중 평균) 및 in-band/out-of-band 계산
을 표준화된 유틸로 구현한다.

## Inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- Task-01 constants/config

## Outputs
- (create) `src/crinv/spectral.py`
- (create) `tests/test_spectral_merge_and_bands.py`

## Constraints
- 30채널 기준 band 정의(B:400–500,G:500–600,R:600–700)를 config/constants에서 관리.
- 301→30 downsample은 블루프린트에서 “training과 동일”이 요구됨.
  - 지금 단계에서는 함수 훅(인터페이스)만 제공하고, 실제 training downsample 함수는 추후 연결 가능하도록 설계.

## Acceptance criteria
- 입력 T shape (`2x2xC` 또는 `Bx2x2xC`)에서 G merge가 올바른 shape로 반환.
- band average 계산이 deterministic.

## Dependencies
- Task-01

## Verification
- `pytest -q tests/test_spectral_merge_and_bands.py`

## Rollback notes
- spectral.py 및 테스트 삭제.
