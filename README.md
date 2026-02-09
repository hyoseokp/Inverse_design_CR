# Color Router Inverse Design (CR Inverse)

Color Router 구조에 대한 end-to-end inverse design 파이프라인(드라이런/모킹 포함)입니다.

- Symmetric `16x16` seed 파라미터화(구조적 `y=x` 대칭 강제)
- Rule-based generator: `16 -> 128`, blur, threshold, STE
- (현재) Mock surrogate로 `128x128 -> 2x2x30` RGGB spectra 생성
- Inverse optimization: multi-start + robustness(σ/τ 샘플링 기대값)
- Top-K export: `npz` 저장 + dashboard용 progress logging(JSONL/NPZ)
- GDS export smoke (gdstk 기반)
- FDTD runner는 “제어흐름 + commit protocol(valid=0/2/1)”만 구현(실제 lumapi 실행 브릿지는 추후)
- FDTD-first ranking(리포트 생성)

## Source Of Truth 문서
- Blueprint/spec: `color-router-inverse-design-blueprint-v1.3.md`
- Repo structure mapping: `repository-structure-mapping.md`
- Task contracts: `tasks/`
- Global common context: `common_context_prompt.md`

## 지금 구현된 범위 (Tasks)
- 구현됨: Task-01 ~ Task-11
- 미구현/플러그인: “실제 forward surrogate 로딩”, “실제 Lumerical lumapi 실행(환경 의존)”

## 실행 순서 (PowerShell)

작업 폴더:
```powershell
cd "C:\Users\연구실\Inverse_design_CR"
```

1) (최초 1회 권장) editable 설치 + 테스트
```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

2) 드라이런 inverse 최적화 실행 (기본값은 빠르게 끝나도록 자동으로 작은 설정으로 override)
```powershell
python -m scripts.run_inverse --dry-run
```

3) 산출물 확인
```powershell
Get-ChildItem data\progress
Get-Content data\progress\metrics.jsonl -TotalCount 3
Get-ChildItem data\candidates
```

## Dashboard (로컬)

Optimization 진행률/손실 곡선/Top-K 구조 스냅샷을 로컬 웹에서 확인할 수 있습니다.

```powershell
python -m scripts.run_dashboard --progress-dir data/progress --port 8501
```

실행하면 콘솔에 로컬 주소가 출력됩니다:
- `http://127.0.0.1:8501/`

Spectrum plot (Top-K를 forward model에 넣어 RGB spectrum을 plot)은 기본 `auto` 입니다.
- `configs/paths.yaml`이 CR_recon 경로/체크포인트를 포함하면 자동으로 켜집니다.
- 강제 옵션: `--surrogate crrecon|mock|none`

4) (옵션) 더 크게/길게 실행
```powershell
python -m scripts.run_inverse --dry-run --n-start 32 --n-steps 200 --topk 8 --robustness-samples 4
```

## 실행 시 도는 루프(요약)

`python -m scripts.run_inverse --dry-run`은 내부적으로 `run_inverse_opt()`를 호출합니다.

1. `A_raw`를 batch(`n_start`)로 초기화(최적화 변수)
2. for step in `n_steps`:
   - `S = seed_from_araw(A_raw)`로 **구조적 대칭 seed** 생성
   - robust MC 샘플링으로 (σ, τ) 여러 쌍 생성(`robustness_samples`)
   - 각 (σ,τ)에 대해:
     - generator로 `u`와 `X_hard/X_ste` 생성(STE로 grad는 `u`를 통해 흐름)
     - surrogate(드라이런은 `MockSurrogate`)로 `T_hat` 예측
     - band metrics 계산 후 loss(ratio/abs + gray/TV) 계산
   - 샘플 평균으로 `E_{σ,τ}[loss]` 구성 → Adam으로 `A_raw` 업데이트
   - progress를 `data/progress/metrics.jsonl`에 append
3. 종료 시 top-K 후보를 뽑아서:
   - `data/candidates/run_*/topk_pack.npz` 저장
   - `data/progress/topk_step-<last>.npz` 스냅샷 저장

## 주요 산출물 경로
- 진행 로그(JSONL): `data/progress/metrics.jsonl`
- Top-K 스냅샷(NPZ): `data/progress/topk_step-*.npz`
- 실행별 Top-K pack: `data/candidates/run_*/topk_pack.npz`

## 안전/승인(중요)
실제 환경 의존/외부 사이드이펙트가 큰 작업은 별도 승인/환경 세팅 후 진행해야 합니다.
- Lumerical lumapi로 실제 FDTD 실행(프로세스 제어/kill/retry 포함)
- GPU 학습(Forward surrogate fine-tune 등)
