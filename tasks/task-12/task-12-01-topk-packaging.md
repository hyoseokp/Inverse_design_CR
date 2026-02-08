# Task 12-01: Surrogate Top-K selection + packaging

## Purpose
Surrogate 기준으로 후보를 넓게(Top-K, 예: 50~200) 뽑고, 이후 단계(FDTD 검증/데이터 append/FT)가 재현 가능하도록 pack 파일로 저장한다.

## Inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- `repository-structure-mapping.md`
- Task-06 outputs (inverse optimization loop)
- Forward model (existing, frozen, GitHub owner=hyoseokp): `hyoseokp/data_CR/code/CR_recon/`
  - `catalog.md`를 먼저 읽고 로딩/입출력 기준 준수

## Outputs
- (create) `src/scripts/run_select_topk.py` (또는 `src/crinv/optimize/select_topk.py` + 스크립트 엔트리)
- (create) `data/inverse_runs/run_xxx/topk_pack.npz`
  - must include: `seed16_topk (K,16,16)`, `struct128_topk (K,128,128 uint8)`, `surrogate_pred (K,2,2,30)`, `metrics_topk`
- (update) `data/progress/` logging (optional) to include Top-K evolution

## Constraints
- K는 config(`configs/inverse.yaml`)로 제어
- Pack 포맷은 다음 단계가 그대로 사용 가능해야 함
- 파일명 규칙/폴더 구조는 mapping 문서 준수

## Acceptance criteria
- Top-K pack이 생성되고, pack만으로 struct/GDS/FDTD 단계로 진행 가능
- pack에 포함된 메트릭으로 Top-M 선정 기준이 명확

## Dependencies
- Task-06

## Verification
- (smoke) `python -m scripts.run_select_topk --help`
- (smoke) pack load 후 shape 확인

## Rollback notes
- `run_xxx/` 폴더 삭제로 원복
