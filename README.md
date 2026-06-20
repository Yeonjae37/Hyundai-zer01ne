**내부 캘리브레이션 및 카메라 포즈 추정 파이프라인**

### 카메라

| 번호 | IP |
|------|----|
| 51 | 192.168.80.51 |
| 52 | 192.168.80.52 |

### 구성

```
calibration/       내부 파라미터(K, dist) 추정
pose_estimation/
├── dlt/           DLT로 3×4 투영행렬 P 추정
└── solvepnp/      solvePnP로 카메라 외부 파라미터(R, t) 추정
```

### 시작

```
cp .env.example .env  # RTSP_TEMPLATE 입력
```

각 폴더 README 참고.