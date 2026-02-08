# Task 07: GDS export (128×128 binary → polygons)

## Purpose
Top-K 구조(128×128 binary)를 GDS로 export한다. 블루프린트 규칙 준수:
- 파일명: `structure_{id:05d}.gds`
- 셀명: `structure_{id:05d}`
- layer: 1 (1:0)

## Inputs
- `color-router-inverse-design-blueprint-v1.3.md`
- `repository-structure-mapping.md` (gds/export_gds.py + polygonize.py 매핑 참고)
- Task-06 exported `struct128` binaries

## Outputs
- (create) `src/crinv/gds_export.py`
- (create) `tests/test_gds_export_smoke.py`
- (create) `data/gds/.gitkeep`

## Constraints
- Polygonization 방식은 결정론적이어야 함.
- Export는 “seed → generator → binary → GDS” 경로를 유지할 수 있도록 인터페이스 설계.

## Acceptance criteria
- smoke 테스트에서 작은 binary 예제(예: 사각형)로 gds 파일이 생성되고, 파일이 비어있지 않음.
- naming/layer 규칙 준수.

## Dependencies
- Task-06

## Verification
- `pytest -q tests/test_gds_export_smoke.py`

## Rollback notes
- gds_export 및 artifacts/gds 삭제.
