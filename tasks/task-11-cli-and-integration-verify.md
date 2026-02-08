# Task 11: CLI + integration verify

## Purpose
사용자가 end-to-end 파이프라인을 실행/검증할 수 있도록 최소 CLI를 제공하고, 통합 검증 리포트를 만든다.

## Inputs
- Task-01..10 outputs

## Outputs
- (create) `src/crinv/cli.py`
- (create) `REPORTS/integration-verify.summary.md`
- (create) `REPORTS/integration-verify.log.md`

## Constraints
- 실제 Lumerical 실행은 환경 의존이므로, 기본 integration verify는:
  - generator/loss/opt의 dry-run
  - gds export smoke
  - commit protocol test
  - dashboard smoke
  위주로 구성.

## Acceptance criteria
- `python -m scripts.run_inverse --help` 또는 `python -m scripts.run_inverse -h` 같은 형태의 CLI 엔트리가 동작.
- integration verify 리포트가 생성되고 PASS/FAIL 기준이 명확.

## Dependencies
- Task-01..10

## Verification
- `python -m scripts.run_inverse --dry-run`

## Rollback notes
- cli.py 및 REPORTS 삭제.
