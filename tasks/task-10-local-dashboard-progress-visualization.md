# Task 10: Local dashboard for optimization progress (loss graph + top-K previews)

## Purpose
Inverse optimization을 돌리는 동안 로컬 브라우저에서 진행상황을 실시간으로 확인할 수 있는 대시보드를 만든다.
- loss curve(시간/step 대비) 그래프
- 현재 top-K 후보들의 구조 preview 이미지(예: 128×128 binary) 및 핵심 metric 표시

## Inputs
- Task-06 (inverse optimization loop) outputs/structures
- Task-03 generator (u, X_hard/struct128)
- Task-01 config (topK, logging interval 등)

## Outputs
- (create) `bot/plans/inverse-design/code/inverse_design/dashboard/app.py` (또는 `server.py`)
- (create) `bot/plans/inverse-design/code/inverse_design/dashboard/state.py`
- (create) `bot/plans/inverse-design/code/inverse_design/dashboard/templates/index.html` (필요 시)
- (create) `bot/plans/inverse-design/code/inverse_design/dashboard/static/` (js/css)
- (create) `bot/plans/inverse-design/code/tests/test_dashboard_smoke.py`
- (update) `bot/plans/inverse-design/code/inverse_design/inverse_opt.py`
  - 진행 로그/스냅샷(예: JSONL/npz/png) 기록 및 대시보드에서 읽을 수 있게 연결

## Constraints
- **Local only**: 외부 공개/배포 금지. 기본은 `127.0.0.1` 바인딩.
- 최적화 루프 성능을 크게 떨어뜨리지 않게, logging interval(예: 매 N step)로 기록.
- 브라우저에서 볼 수 있도록 이미지 생성은 lightweight(예: PNG/Canvas) 방식.

### Data contract (Task-06과 인터페이스 고정)
대시보드는 아래 파일만 읽어서 동작해야 한다(최적화 프로세스와 느슨하게 결합):
- `code/artifacts/progress/metrics.jsonl` (append-only)
- `code/artifacts/progress/topk_step-<step>.npz` (periodic)
- (optional) `code/artifacts/progress/previews/*.png`

- 의존성 최소화:
  - 옵션 A: Streamlit
  - 옵션 B: FastAPI + simple JS chart
  - (선호) 설치/실행이 쉬운 쪽으로 선택하되, `pyproject.toml`에 명시.

## Acceptance criteria
- `python -m inverse_design.dashboard.app` 실행 시 로컬 URL이 출력되고 접속 가능.
- loss graph가 step 진행에 따라 업데이트됨.
- top-K preview가 최소 K=5 이상 표시됨.
- 최적화가 끝나도 로그 파일 기반으로 과거 결과를 재조회 가능.

## Dependencies
- Task-01, Task-03, Task-06

## Verification
- `pytest -q tests/test_dashboard_smoke.py`
- 수동: (로컬) 대시보드 실행 후 브라우저 접속 확인

## Rollback notes
- dashboard 폴더 및 inverse_opt 연결 코드 제거.
