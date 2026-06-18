# Camera Calibration

## 세팅

```
pip install -r requirements.txt
```

`.env.example`을 복사해서 `.env` 만들고 RTSP URL 입력

```
cp .env.example .env
```

## camera number

| 번호 | IP |
|------|----|
| 51 | 192.168.80.51 |
| 52 | 192.168.80.52 |

## 사용법

**1. 캡처** — 스페이스바로 프레임 저장, Q로 종료
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
