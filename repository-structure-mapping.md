# Color Router Inverse Design – Repository Structure Mapping

이 문서는 **Color Router Inverse Design – Final Blueprint (v1.3)**를 실제 **코드 레포지토리 구조**로 1:1 매핑한 구현 설계서이다.  
Inverse 최적화, rule-based 구조 생성, GDS export, Lumerical FDTD 검증, 결과 랭킹까지를 **재현 가능하고 확장 가능한 코드 구조**로 정의한다.

---

## 1. 최상위 레포 구조

```
color-router-inv/
├─ README.md
├─ pyproject.toml            # 또는 requirements.txt
├─ .gitignore
│
├─ configs/
│  ├─ inverse.yaml           # inverse 최적화 하이퍼파라미터
│  ├─ fdtd.yaml              # FDTD 실행/타임아웃/청크 설정
│  └─ paths.example.yaml     # 로컬 경로 예시 (깃에는 예시만)
│
├─ src/
│  ├─ crinv/
│  │  ├─ __init__.py
│  │  │
│  │  ├─ models/
│  │  │  ├─ forward_surrogate.py   # 128x128 → 2x2x30 wrapper (frozen)
│  │  │  └─ ensemble.py            # (선택) forward ensemble
│  │  │
│  │  ├─ generator/
│  │  │  ├─ seed_param.py          # A_raw → sigmoid → (A+Aᵀ)/2 (대칭 강제)
│  │  │  ├─ rule_base.py           # upsample / blur / threshold / STE
│  │  │  └─ kernels.py             # Gaussian kernel precompute (σ set)
│  │  │
│  │  ├─ spectra/
│  │  │  ├─ wavelength.py          # λ grid, trapezoid weights, band masks
│  │  │  └─ downsample.py          # 301 → 30 (학습과 동일 함수)
│  │  │
│  │  ├─ objectives/
│  │  │  ├─ loss_rgb.py            # D_b / O_b, ratio+abs, gray/TV
│  │  │  └─ metrics.py             # contrast, robustness stats
│  │  │
│  │  ├─ optimize/
│  │  │  ├─ inverse_search.py      # multi-start batched Adam + robustness MC
│  │  │  ├─ select_topk.py         # surrogate 기준 top-K 선정
│  │  │  └─ checkpoints.py         # 결과/재개 관리
│  │  │
│  │  ├─ gds/
│  │  │  ├─ export_gds.py          # struct128 → GDS (cell/file 규칙 고정)
│  │  │  └─ polygonize.py          # binary grid → polygons
│  │  │
│  │  ├─ fdtd/
│  │  │  ├─ env.py                 # Lumerical 경로 설정 + norm_path
│  │  │  ├─ scripts.py             # gdsimport script, EXTRACT_SCRIPT
│  │  │  ├─ runner.py              # session/timeout/kill/retry
│  │  │  ├─ memmap_io.py           # spectra memmap, valid_mask, progress
│  │  │  └─ validate_topk.py       # TopK → seed → struct → GDS → FDTD
│  │  │
│  │  └─ reports/
│  │     ├─ make_report.py         # surrogate vs FDTD 비교 리포트
│  │     └─ plots.py               # (선택) 시각화
│  │
│  └─ scripts/
│     ├─ run_inverse.py            # CLI: inverse 실행
│     ├─ run_export_gds.py         # CLI: topK → GDS 생성
│     ├─ run_fdtd_validate.py      # CLI: GDS → FDTD 검증 (재개 가능)
│     └─ run_rank_final.py         # CLI: 최종 랭킹
│
├─ data/                          # (깃 제외 권장)
│  ├─ forward_ckpt/               # forward surrogate 체크포인트
│  ├─ inverse_runs/               # inverse 결과(run별)
│  ├─ gds_valid/                  # 검증용 GDS
│  ├─ fdtd_results/               # FDTD spectra / valid / progress
│  └─ wavelength/                 # wavelength grid 저장
│
└─ docs/
   ├─ blueprint.md                # Final Blueprint(v1.3)
   └─ fdtd_template_notes.md      # Trans_1/2/3 의미, sign convention
```

---

## 2. 핵심 모듈 책임 매핑

### 2.1 Seed 파라미터화 (대칭 강제 – 최우선 규칙)
- **파일**: `generator/seed_param.py`
- **역할**:
  - optimizer가 업데이트하는 변수는 `A_raw`뿐
  - 모든 forward/backward에서
    ```python
    A = sigmoid(A_raw)
    S = 0.5 * (A + A.T)
    ```
  - y=x 대각선 대칭을 구조적으로 보장

---

### 2.2 Rule-based Generator (불변)
- **파일**: `generator/rule_base.py`
- **역할**:
  - seed → 구조 매핑의 단일 진실(source of truth)
  - inverse, 학습 데이터 생성, FDTD 검증에서 **동일 코드 재사용**
  - STE는 inverse에서만 사용, 검증/FDTD에서는 hard threshold만 사용

---

### 2.3 Loss / Metrics
- **파일**:
  - `objectives/loss_rgb.py`
  - `objectives/metrics.py`
- **역할**:
  - G 픽셀 병합(R,G,B 3채널 평가)
  - ratio 기반 분리 + 절대 효율 보조
  - gray/TV 정규화

---

### 2.4 Inverse Optimization
- **파일**: `optimize/inverse_search.py`
- **역할**:
  - multi-start batched optimization
  - robustness MC (σ, τ jitter)
  - surrogate 기준 top-K 후보 생성

---

### 2.5 GDS Export
- **파일**: `gds/export_gds.py`
- **역할**:
  - `struct128 (numpy)` → `structure_{id:05d}.gds`
  - 셀명/파일명 규칙을 FDTD import와 정확히 일치

---

### 2.6 FDTD Validation (정석 루트)
- **파일**: `fdtd/validate_topk.py`
- **역할**:
  - top-K torch → seed16 numpy
  - rule-based generator로 struct128 재생성 (nominal σ₀, τ₀)
  - GDS export → lumapi FDTD 실행
  - spectra(301) memmap 저장 + 재개 가능

---

### 2.7 최종 리포트 / 랭킹
- **파일**: `reports/make_report.py`
- **역할**:
  - FDTD(301→30) vs surrogate(30) 비교
  - band metric, contrast, error 요약

---

## 3. CLI 실행 플로우

### 3.1 Inverse 탐색
```bash
python -m scripts.run_inverse --config configs/inverse.yaml
```

### 3.2 Top-K GDS 생성
```bash
python -m scripts.run_export_gds \
  --pack data/inverse_runs/run_xxx/topk_pack.npz \
  --out data/gds_valid/
```

### 3.3 FDTD 검증 (재개 가능)
```bash
python -m scripts.run_fdtd_validate --config configs/fdtd.yaml
```

### 3.4 최종 랭킹
```bash
python -m scripts.run_rank_final \
  --pack data/inverse_runs/run_xxx/topk_pack.npz \
  --fdtd data/fdtd_results/inv_topk_spectra.npy
```

---

## 4. 레포 설계 불변 조건 (Invariant)

1. Seed 대칭은 **파라미터화로만** 강제
2. Rule-based generator는 단 하나
3. FDTD 검증은 항상 `seed → generator → GDS` 경로
4. Forward surrogate는 inverse 중 frozen
5. 최종 선택 기준은 FDTD

---

**이 문서는 Blueprint(v1.3)를 실제 코드 레포지토리로 구현하기 위한 공식 매핑 문서이다.**
