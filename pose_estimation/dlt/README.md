# DLT Projection Matrix

ArUco 마커로 3×4 투영행렬 P를 추정. K 없이 순수 선형 방법으로 품.

## 실행

```
python3 dlt_projection.py --cam 52
```

## 출력

| 파일 | 내용 |
|------|------|
| `projection_matrix.npy` | 3×4 P 행렬 |
| `dlt_result.jpg` | 검출 코너(초록) vs 재투영 코너(빨강) |