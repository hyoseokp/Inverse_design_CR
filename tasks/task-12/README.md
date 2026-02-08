# Task 12 (Folder, Optional): Active loop (Top-K → Top-M FDTD validate → dataset append → forward fine-tune → inverse re-run)

## 개요
Forward surrogate가 imperfect할 때 inverse는 surrogate의 오차/편향을 “역이용”하는 방향으로 최적화가 밀릴 수 있다.
이를 정면으로 해결하기 위해, surrogate로 넓게 후보를 뽑고(Top-K) 그중 일부만 FDTD로 검증(Top-M)한 뒤,
검증 데이터를 데이터셋에 추가하여 surrogate를 fine-tune하고, 다시 inverse를 재실행하는 루프를 구성한다.

## 실행 순서
1) `task-12-01-topk-packaging.md`
2) `task-12-02-topm-fdtd-validate-and-dataset-append.md`
3) `task-12-03-forward-finetune-and-rerun-inverse.md`

## 산출물 핸드오프(요약)
- Top-K pack: `data/inverse_runs/run_xxx/topk_pack.npz`
- Top-M FDTD spectra + progress: `data/fdtd_results/run_xxx/`
- Dataset append bundle: `data/dataset/append/run_xxx/`
- Fine-tuned surrogate checkpoints: `data/forward_ckpt/ft_run_xxx/`

## 불변 조건(재강조)
- Seed symmetry는 파라미터화로만 강제
- Generator는 단 하나(source of truth)
- FDTD 검증은 항상 `seed → generator → GDS → FDTD`
- 최종 랭킹/선택은 FDTD 기준
