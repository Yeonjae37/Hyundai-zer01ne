# Camera Calibration

### 카메라

| 번호 | IP |
|------|----|
| 51 | 192.168.80.51 |
| 52 | 192.168.80.52 |

## 실행

**1. 프레임 수집** — space bar로 프레임 저장, Q로 종료
```
python3 capture.py <camera number>
```

**2. 캘리브레이션** — `frames/<번호>/` 이미지로 캘리브레이션
```
python3 calibrate.py <camera number>
```

**3. 왜곡 보정 확인** — 보정 전/후 나란히 보기
```
python3 view.py <camera number>
```
