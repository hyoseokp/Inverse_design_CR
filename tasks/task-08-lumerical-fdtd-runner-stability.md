# Task 08: Lumerical FDTD runner (lumapi) + stability rules

## Purpose
블루프린트의 정석 FDTD 검증 루트를 구현한다:
- (중요) **FDTD 검증은 반드시 seed → generator → GDS**를 거친다.
- Windows + Lumerical v241 환경에서 lumapi를 통해 템플릿 fsp 로딩 후 GDS import → run → spectra extraction.
- 안정성 규칙(Chunk/timeout/zombie kill/retry/2-phase commit) 준수.

## Inputs
- `bot/plans/inverse-design/color-router-inverse-design-blueprint-v1.3.md`
- `bot/plans/inverse-design/repository-structure-mapping.md` (fdtd/* 모듈 책임/CLI 플로우 매핑 참고)
- Task-07 GDS export outputs
- Lumerical template file path (user/environment provided): `air_SiN_2um_NA.fsp`
- **Reference (existing code, GitHub owner=hyoseokp):** `hyoseokp/data_CR/data_gen.ipynb`
  - 여기서 `hyoseokp`는 효석의 GitHub 계정(소유자) 이름.
  - FDTD를 Python으로 제어(lumapi/FDTD 템플릿/GDS import/스펙트럼 추출)하는 실전 예제가 들어있으니,
    `fdtd_runner.py`/`fdtd_scripts.py` 설계 시 이 노트북의 패턴을 참고할 것.

## Outputs
- (create) `bot/plans/inverse-design/code/inverse_design/fdtd_runner.py`
- (create) `bot/plans/inverse-design/code/inverse_design/fdtd_scripts.py` (gdsimport/extract 스크립트 템플릿)
- (create) `bot/plans/inverse-design/code/inverse_design/commit_protocol.py` (valid=0/2/1)
- (create) `bot/plans/inverse-design/code/artifacts/fdtd_runs/.gitkeep`
- (create) `bot/plans/inverse-design/code/tests/test_commit_protocol.py`

## Constraints
- Approval required: 실제 Lumerical 실행/프로세스 kill 같은 외부/환경 의존 행동은 실행 전 사용자 승인 필요.
- Runner는 chunk-based batch 실행을 지원해야 함.
- Retry 정책 및 타임아웃 파라미터는 config로 노출.

## Acceptance criteria
- commit protocol(valid=0/2/1)이 코드로 구현되고 테스트로 검증됨.
- 실제 lumapi 없이도(모킹) runner 제어 흐름은 테스트 가능.

## Dependencies
- Task-07

## Verification
- `pytest -q tests/test_commit_protocol.py`

## Rollback notes
- fdtd_runner 관련 파일 삭제.
