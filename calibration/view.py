import cv2
import numpy as np
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

    cam_num    = sys.argv[1]
    params_path = os.path.join("params", cam_num, "intrinsics.npz")

    if not os.path.exists(params_path):
        print(f"Calibration not found: {params_path}")
        print("Run calibrate.py first.")
        sys.exit(1)

    d = np.load(params_path)
    K, dist, img_size = d["K"], d["dist"], tuple(d["img_size"])

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, img_size, alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, img_size, cv2.CV_16SC2)
    x, y, w, h = roi

    url = RTSP_TEMPLATE.format(cam=cam_num)
    print(f"[CAM{cam_num}] Connecting... {url}")
    cap = RTSPReader(url)
    print(f"[CAM{cam_num}] Connected  |  Q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        if w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]
            undistorted = cv2.resize(undistorted, (frame.shape[1], frame.shape[0]))

        cv2.putText(frame,       "ORIGINAL",    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 220), 2)
        cv2.putText(undistorted, "UNDISTORTED", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0),  2)

        cv2.imshow(f"CAM{cam_num}  |  before / after", np.hstack([frame, undistorted]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
