# SolvePnP Camera Pose

ArUco 마커로 카메라 외부 파라미터(R, t)를 추정. K는 calibration 결과를 사용.

## 실행

```
python3 solvepnp_pose.py --cam 52
```

## 출력

| 파일 | 내용 |
|------|------|
| `rvec.npy` | 회전 벡터 (Rodrigues) |
| `tvec.npy` | 이동 벡터 (cm) |
| `solvepnp_result.jpg` | 재투영 결과 + XYZ 축 시각화 |