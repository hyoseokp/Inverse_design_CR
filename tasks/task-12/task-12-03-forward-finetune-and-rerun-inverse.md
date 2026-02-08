# Task 12-03: Forward fine-tune + re-run inverse (active loop)

## Purpose
FDTD로 검증된 Top-M 데이터를 데이터셋에 추가한 뒤, forward surrogate를 fine-tune하고,
새 surrogate로 inverse를 재실행하여 “모델 오차 역이용” 문제를 완화한다.

## Inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- `repository-structure-mapping.md`
- Dataset append bundle: `data/dataset/append/run_xxx/`
- Forward model source (GitHub owner=hyoseokp): `hyoseokp/data_CR/code/CR_recon/`
  - `catalog.md` 먼저 읽고 학습/로딩 규칙 확인
- (Optional) training scripts from CR_recon or separate training repo

## Outputs
- (create) `src/crinv/models/finetune_forward.py` (interface/stub OK if training code external)
- (create) `src/scripts/run_finetune_forward.py`
- (create) `data/forward_ckpt/ft_run_xxx/` (new checkpoints)
- (create) `REPORTS/forward_finetune.summary.md`
  - before/after error (Top-M on FDTD)
  - calibration plots (optional)
- (re-run) inverse outputs under `data/inverse_runs/run_xxx_ft/`

## Constraints
- **Approval required**: 실제 학습/파인튜닝 실행은 사용자 승인 후 진행
- Fine-tune 후에도 surrogate는 inverse 동안 frozen
- 실행 재현성을 위해 config와 dataset snapshot을 함께 보관

## Acceptance criteria
- fine-tuned checkpoint 생성
- Top-M에 대한 surrogate error가 의미 있게 감소(정량 지표 보고)
- 새 checkpoint로 inverse를 재실행할 수 있음

## Dependencies
- Task-12-02

## Verification
- (smoke) new ckpt loadable
- (report) `REPORTS/forward_finetune.summary.md` 생성

## Rollback notes
- `data/forward_ckpt/ft_run_xxx/` 및 `run_xxx_ft/` 폴더 제거
