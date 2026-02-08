# Task 01: Project scaffold + config schema

## Purpose
Inverse-design 프로젝트를 `inverse-design/` 폴더 안에서 **재현 가능한 코드/데이터/결과 구조**로 세팅하고, 블루프린트(v1.3)의 파라미터(σ/τ/밴드 정의 등)를 코드로 표현 가능한 **config 스키마**로 정리한다.

## Inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- `repository-structure-mapping.md` (레포 구조 매핑 문서)
- **Reference (existing code, GitHub owner=hyoseokp):** `hyoseokp/data_CR/data_gen.ipynb`
  - 여기서 `hyoseokp`는 효석의 GitHub 계정(소유자) 이름.
  - FDTD 실행/제어 관련 코드가 있으므로, Task-08에서 재사용/정합이 가능하도록
    config에서 템플릿 경로, gds import 파라미터, extract script 등을 표현할 수 있게 고려.
- **Forward model (existing, frozen, GitHub owner=hyoseokp):** `hyoseokp/data_CR/code/CR_recon/`
  - 128×128 구조 → 2×2×30 surrogate 모델 및 파라미터가 저장되어 있음
  - forward model 정보를 확인하려면 폴더 내 `catalog.md`를 먼저 읽고 그 기준을 따른다.
  - config에서 surrogate 경로/체크포인트/출력채널(30) 등을 표현할 수 있게 고려.

## Outputs (create)
- `README.md`
- `pyproject.toml` (또는 `requirements.txt` — 선택 1개)
- `.gitignore`
- `configs/paths.example.yaml`
- `configs/inverse.yaml` (skeleton)
- `configs/fdtd.yaml` (skeleton)
- `src/crinv/__init__.py`
- `src/crinv/config.py` (dataclass/pydantic 중 택1)
- `src/crinv/constants.py` (RGB band ranges, eps 등)
- `tests/test_config_smoke.py`

## Constraints
- 코드 산출물은 repository-structure-mapping.md의 구조를 따른다 (`src/crinv/**`, `configs/**`, `tests/**`).
- 스펙(블루프린트)과 다른 임의 변경 금지.

## Acceptance criteria
- config에 블루프린트의 핵심 파라미터가 모두 표현된다:
  - seed size(16), struct size(128)
  - blur σ set
  - threshold τ0, Δτ
  - band ranges (B/G/R)
  - output channels (30 or 301) 선택
  - epsilon, loss weights
  - optimization params (N_start, N_steps, Adam lr 등)
- `pytest -q`로 `test_config_smoke.py` 통과.

## Dependencies
- 없음

## Verification
- (로컬) `python -c "from crinv.config import InverseDesignConfig; print(InverseDesignConfig())"`
- (테스트) `pytest -q`

## Rollback notes
- 생성된 파일들을 git에서 되돌리면 원복 가능.
