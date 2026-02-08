# Color Router Inverse Design – Final Blueprint (v1.3)

본 문서는 **컬러 라우터(color router)** 구조를 위한 **Inverse Design 전체 설계도**이다.\
Forward surrogate 학습 완료 상태에서 시작하여, **대각선 대칭 seed 기반 최적화 → rule-based 구조 생성 → GDS export → Lumerical FDTD 검증**까지의 **완전한 end-to-end 파이프라인**을 구현 가능 수준으로 명시한다.

---

## 0. 문제 정의 및 전제

### 픽셀 배치

```
[[ R ,  G ],
 [ G ,  B ]]
```

- y = x 대각선 대칭 구조
- 두 Green 픽셀(G1, G2)은 **물리적으로 항상 동일**

### 목표 스펙트럼 대역

- **B 픽셀**: 400–500 nm → 최대 투과, 그 외 최소
- **G 픽셀**: 500–600 nm → 최대 투과, 그 외 최소
- **R 픽셀**: 600–700 nm → 최대 투과, 그 외 최소

### 데이터 파이프라인(고정)

```
16×16 real seed  →  rule-based generator  → 128×128 binary structure
                                         → forward surrogate
                                         → 2×2×30 (또는 301) RGGB spectrum
```

---

## 1. Forward Model 상태 (가정)

- Forward surrogate `f_θ`는 이미 학습 완료
- 입력: `X ∈ {0,1}^{128×128}`
- 출력: `T̂ ∈ R^{2×2×30}` (또는 `2×2×301`)
- 학습 데이터는 **Lumerical FDTD + rule-based 구조 생성**으로 생성됨

Forward 모델은 **inverse 과정에서 고정(frozen)** 되어 평가기로만 사용된다.

---

## 2. Inverse Design 핵심 원칙

### 2.1 최적화 변수

- **직접 최적화되는 변수는 seed가 아니라 원시 변수**

```
A_raw ∈ R^{16×16}
```

### 2.2 대각선 대칭 강제 (필수 규칙)

대칭은 loss나 post-processing이 아닌 **파라미터화 단계에서 구조적으로 강제**한다.

#### Seed 생성 규칙 (모든 forward / backward에서 동일)

```
A = sigmoid(A_raw)
S = (A + Aᵀ) / 2
```

- optimizer는 **오직 ****A\_raw****만 업데이트**
- `S`는 항상 y=x 대각선 대칭
- backpropagation 시 gradient도 자동으로 대칭 유지

> ❌ 금지: symmetry loss, step 후 symmetrize, upper-triangle hack

---

## 3. Rule-based Structure Generator (고정)

Inverse와 FDTD 검증 모두 **동일한 rule-based generator**를 사용해야 한다.

### 3.1 Generator 파이프라인

```
S (16×16, symmetric)
 → bilinear upsample
 → U (128×128)
 → gaussian blur (σ)
 → u
 → threshold (τ)
 → X (128×128 binary)
```

### 3.2 Threshold 처리 (Inverse 내부)

- **Forward**: hard threshold (binary)
- **Backward**: STE (Straight-Through Estimator)

```
X_hard = (u > τ)
X_ste  = X_hard + u - stopgrad(u)
```

- Forward surrogate 입력은 항상 binary
- Gradient는 u를 통해 흐름

### 3.3 Robust Design

Inverse 중에는 공정 변동을 반영하기 위해 σ, τ를 랜덤 샘플링:

- σ ∼ discrete set (예: {0.8, 1.0, 1.2})
- τ ∼ U(τ₀ − Δτ, τ₀ + Δτ)

Loss는 기대값 형태로 최적화:

```
L = E_{σ,τ}[ L_spec + L_reg ]
```

---

## 4. Loss Function (RGB 기준, G 병합)

### 4.1 스펙트럼 처리

- Green은 항상 평균 사용:

```
T_G = (T_01 + T_10) / 2
```

### 4.2 Band Average (30채널 기준)

각 band b ∈ {R,G,B}에 대해

```
A_{c,b} = weighted mean of T_c over band b
```

### 4.3 In-band / Out-of-band 정의

```
D_b = A_{b,b}
O_b = mean of A_{c,b} for c ≠ b
```

### 4.4 Spectral Loss

- Separation (ratio-based):

```
L_ratio = Σ_b log((O_b + ε)/(D_b + ε))
```

- Absolute efficiency support:

```
L_abs = Σ_b (1 − D_b)
```

```
L_spec = w_ratio · L_ratio + w_abs · L_abs
```

### 4.5 Regularization

- Gray penalty (threshold robustness):

```
L_gray = mean(u · (1 − u))
```

- TV penalty (minimum feature size):

```
L_tv = total_variation(u)
```

```
L_reg = w_gray · L_gray + w_tv · L_tv
```

---

## 5. Inverse Optimization Loop

### 5.1 Multi-start Strategy

- Batch optimization of multiple seeds
- Typical settings:

```
N_start = 100–500
N_steps = 1000–5000
Optimizer = Adam
```

### 5.2 Canonical Optimization Loop

```python
A_raw = Parameter([B,16,16])

for step in range(N_steps):
    A = sigmoid(A_raw)
    S = 0.5 * (A + A.T)   # symmetry enforced here

    U = upsample(S)
    u = gaussian_blur(U, σ)
    X_ste = STE_threshold(u, τ)

    T_hat = forward_surrogate(X_ste)
    loss = E_{σ,τ}[compute_loss(T_hat, u)]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 5.3 Inverse Output (Top-K)

For each candidate k:

- `seed16[k]` (numpy)
- `struct128[k]` (binary, nominal σ₀, τ₀)
- surrogate prediction + metrics

---

## 6. FDTD Validation (정석 루트)

**FDTD 검증은 반드시 seed로 되돌아가 rule-based generator를 다시 거친다.**

### 6.1 Validation Pipeline (고정)

```
Top-K torch seeds
 → numpy seed16
 → rule-based generator (nominal σ₀, τ₀)
 → 128×128 numpy binary
 → GDS export
 → lumapi FDTD run
 → spectra (301)
```

### 6.2 Torch → Numpy Seed

```
seed16_np = seed16_torch.detach().cpu().numpy()
seed16_np = 0.5*(seed16_np + seed16_np.T)
```

### 6.3 GDS Export 규칙

- 파일명: `structure_{id:05d}.gds`
- 셀명: `structure_{id:05d}`
- Layer: 1 (LAYER\_MAP = "1:0")
- Binary grid → polygonization

---

## 7. Lumerical FDTD Execution 규칙

### 7.1 Environment

- Windows + Lumerical v241
- `lumapi.FDTD(hide=True)`
- Template file: `air_SiN_2um_NA.fsp`

### 7.2 Execution Steps (per structure)

1. `fdtd.switchtolayout()`
2. `fdtd.eval(gdsimport_script)`
3. check `import_ok`
4. `fdtd.run()`
5. `fdtd.eval(EXTRACT_SCRIPT)`
6. extract `T1, T2, T3, f_vec`

### 7.3 Stability Rules (필수)

- Chunk-based sessions
- Stage-wise timeout
- Zombie fdtd process kill
- Retry on failure
- 2-phase commit:

```
valid = 0 (not run)
valid = 2 (written, not committed)
valid = 1 (committed)
```

---

## 8. Post-Processing & Final Selection

### 8.1 Spectral Processing

- Apply same sign convention as training (e.g. `T = -T`)
- Convert f → wavelength (nm)
- Downsample 301 → 30 using **same function as training**

### 8.2 Metrics

- Band averages D\_b, O\_b
- Contrast ratios
- Robustness (optional jitter re-eval)

### 8.3 Final Ranking

- Rank by **FDTD metrics**, not surrogate
- Surrogate vs FDTD error reported
- Top-M selected as final designs

---

## 9. Optional Active Loop

1. Add validated (X, T\_FDTD) to dataset
2. Fine-tune forward surrogate
3. Re-run inverse optimization

---

## 10. Design Invariants (절대 위반 금지)

1. Seed symmetry is **structural**, not penalized
2. Same rule-based generator for training, inverse, and FDTD
3. FDTD validation always goes through **seed → generator → GDS**
4. Forward surrogate is frozen during inverse
5. Final ranking uses **FDTD**, not surrogate

---

**This document defines the final, canonical inverse-design pipeline for the color router problem.**
