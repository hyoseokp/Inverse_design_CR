# Task 12-02: Top-M FDTD validation + dataset append

## Purpose
Top-K 중에서 Top-M(예: 10~50)을 골라 FDTD로 검증하고, 그 결과를 데이터셋에 추가(append)한다.

## Inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- `repository-structure-mapping.md`
- Top-K pack: `data/inverse_runs/run_xxx/topk_pack.npz`
- Reference (existing code, GitHub owner=hyoseokp): `hyoseokp/data_CR/data_gen.ipynb`
  - lumapi/FDTD 제어 패턴 참고
- Lumerical template (local path, user-provided)

## Outputs
- (create) `src/scripts/run_fdtd_validate_topm.py` (또는 `src/crinv/fdtd/validate_topk.py` 확장)
- (create) `data/fdtd_results/run_xxx/`
  - spectra(301) 저장 + valid/progress(재개 가능)
- (create) `data/dataset/append/run_xxx/`
  - `X_struct128.npy` (M,128,128 uint8)
  - `T_fdtd.npy` (M,2,2,301) 또는 다운샘플 포함 버전
  - metadata.json (σ0, τ0, template hash, timestamps)

## Constraints
- **Approval required**: 실제 lumapi/FDTD 실행은 사용자 승인 후 진행
- 반드시 `seed → generator → GDS → FDTD` 경로 유지
- 2-phase commit(valid=0/2/1) + retry/timeout 정책 유지

## Acceptance criteria
- Top-M에 대해 FDTD 결과가 생성되고, 재시작 시 이어서 실행 가능
- append 데이터셋 번들이 생성되어 fine-tune 입력으로 사용 가능

## Dependencies
- Task-07, Task-08, Task-12-01

## Verification
- (smoke) 결과 폴더에 progress/valid mask 존재
- (smoke) dataset append bundle load 가능

## Rollback notes
- `data/fdtd_results/run_xxx/` 및 `data/dataset/append/run_xxx/` 제거
