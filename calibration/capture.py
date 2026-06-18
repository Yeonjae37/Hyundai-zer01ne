import cv2
import os
import sys
import time
import threading
from dotenv import load_dotenv

load_dotenv()
RTSP_TEMPLATE = os.environ["RTSP_TEMPLATE"]

class RTSPReader:
    def __init__(self, url):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {url}")
        self._cap = cap
        self._frame = None
        self._lock = threading.Lock()
        self._stop = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self._stop:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._stop = True
        self._cap.release()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cam_num = sys.argv[1]
    url     = RTSP_TEMPLATE.format(cam=cam_num)
    save_dir = os.path.join("frames", cam_num)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[CAM{cam_num}] : {url}")
    cap = RTSPReader(url)
    print(f"[CAM{cam_num}] connected..  |  SPACE=capture  Q=quit\n")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        disp = frame.copy()
        cv2.putText(disp, f"CAM{cam_num}  saved: {count}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)
        cv2.imshow(f"CAM{cam_num}", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            ts   = int(time.time() * 1000)
            path = os.path.join(save_dir, f"{ts}.png")
            cv2.imwrite(path, frame)
            count += 1
            print(f"  [{count}] save → {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal {count} saved")


if __name__ == "__main__":
    main()
