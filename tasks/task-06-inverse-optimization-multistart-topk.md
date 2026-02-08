# Task 06: Inverse optimization loop (multi-start + top-K export)

## Purpose
블루프린트의 canonical optimization loop를 구현:
- `A_raw` 파라미터
- symmetry seed 생성
- generator(σ/τ 샘플링 + STE)
- frozen surrogate 평가
- loss 계산
- Adam 업데이트
또한 multi-start(batch)로 여러 초기값을 동시에 최적화하고, top-K 결과를 파일로 저장한다.

## Inputs
- `bot/plans/inverse-design/color-router-inverse-design-blueprint-v1.3.md`
- `bot/plans/inverse-design/repository-structure-mapping.md` (optimize/inverse_search.py 매핑 참고)
- Task-02,03,04,05 outputs
- **Forward model (existing, frozen, GitHub owner=hyoseokp):** `hyoseokp/data_CR/code/CR_recon/`
  - 여기서 `hyoseokp`는 효석의 GitHub 계정(소유자) 이름.
  - 128×128 구조 → 2×2×30 surrogate 모델 및 파라미터가 저장되어 있음
  - 모델/입출력/로드 방법은 폴더 내 `catalog.md`를 먼저 읽고 그 기준을 따른다.

## Outputs
- (create) `bot/plans/inverse-design/code/inverse_design/inverse_opt.py`
- (create) `bot/plans/inverse-design/code/inverse_design/surrogate_interface.py` (forward surrogate를 plug-in 형태로)
- (create) `bot/plans/inverse-design/code/inverse_design/artifacts.py` (seed/struct/metrics 저장 포맷)
- (create) `bot/plans/inverse-design/code/inverse_design/progress_logger.py` (dashboard용 진행 로그/스냅샷 기록)
- (create) `bot/plans/inverse-design/code/tests/test_inverse_opt_dryrun.py` (mock surrogate로 2~3 step 동작)
- (create) `bot/plans/inverse-design/code/artifacts/candidates/.gitkeep`
- (create) `bot/plans/inverse-design/code/artifacts/progress/.gitkeep`

## Constraints
- Forward surrogate는 inverse 동안 frozen.
- Final output은 최소:
  - seed16 numpy
  - struct128 binary(nominal σ0, τ0)
  - surrogate prediction + band metrics
- Randomness는 seed로 통제 가능해야 함.

### Dashboard logging hook (필수)
최적화 도중 대시보드가 읽을 수 있도록 **파일 기반**으로 진행상황을 기록한다.

권장 포맷(가볍고 재현 가능):
- Metrics log (append-only JSONL):
  - `bot/plans/inverse-design/code/artifacts/progress/metrics.jsonl`
  - 최소 필드: `ts`, `step`, `seed_id(or batch idx)`, `loss_total`, `loss_spec`, `loss_reg`, `D_R`, `D_G`, `D_B`, `O_R`, `O_G`, `O_B`
- Top-K snapshot (periodic, overwrite OK or versioned):
  - `bot/plans/inverse-design/code/artifacts/progress/topk_step-<step>.npz`
    - arrays: `struct128_topk` (K,128,128 uint8), `seed16_topk` (K,16,16 float32), `metrics_topk`
- Preview images (optional but helpful):
  - `bot/plans/inverse-design/code/artifacts/progress/previews/topk_step-<step>_k-<k>.png`

성능 제약:
- `log_every_n_steps`를 config로 노출하고 기본값을 10~50 사이로.
- PNG 생성은 optional(끄면 npz+jsonl만으로도 dashboard가 렌더링 가능).

## Acceptance criteria
- mock surrogate로도 end-to-end loop가 돌아가며 loss가 감소하는 방향(완벽할 필요는 없지만 업데이트는 발생).
- top-K export 파일이 생성됨.

## Dependencies
- Task-02..05

## Verification
- `pytest -q tests/test_inverse_opt_dryrun.py`

## Rollback notes
- inverse_opt 및 artifacts 폴더 삭제.
