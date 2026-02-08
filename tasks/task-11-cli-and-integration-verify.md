# Task 11: CLI + integration verify

## Purpose
사용자가 end-to-end 파이프라인을 실행/검증할 수 있도록 최소 CLI를 제공하고, 통합 검증 리포트를 만든다.

## Inputs
- Task-01..10 outputs

## Outputs
- (create) `bot/plans/inverse-design/code/inverse_design/cli.py`
- (create) `bot/plans/inverse-design/code/REPORTS/integration-verify.summary.md`
- (create) `bot/plans/inverse-design/code/REPORTS/integration-verify.log.md`

## Constraints
- 실제 Lumerical 실행은 환경 의존이므로, 기본 integration verify는:
  - generator/loss/opt의 dry-run
  - gds export smoke
  - commit protocol test
  - dashboard smoke
  위주로 구성.

## Acceptance criteria
- `python -m inverse_design.cli --help`가 동작.
- integration verify 리포트가 생성되고 PASS/FAIL 기준이 명확.

## Dependencies
- Task-01..10

## Verification
- `python -m inverse_design.cli --dry-run`

## Rollback notes
- cli.py 및 REPORTS 삭제.
