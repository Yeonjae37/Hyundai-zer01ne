import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

RTSP_TEMPLATE   = os.environ["RTSP_TEMPLATE"]
LAYOUT_YAML     = Path(__file__).parent.parent / "marker_layout.yaml"
CALIBRATION_DIR = Path(__file__).parent.parent.parent / "calibration" / "params"
ARUCO_DICT      = cv2.aruco.DICT_4X4_100
OUT_DIR         = Path(__file__).parent


def grab_frame(url: str) -> np.ndarray:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        sys.exit("스트림 연결 실패.")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit("프레임 수신 실패.")
    return frame


def load_layout(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    return {
        int(mid): np.array(info["corners_3d"], dtype=np.float64)
        for mid, info in data["markers"].items()
    }


def load_intrinsics(cam: str):
    path = CALIBRATION_DIR / cam / "intrinsics.npz"
    if not path.exists():
        sys.exit(f"캘리브레이션 없음: {path}\ncalibrate.py {cam} 먼저 실행하세요.")
    d = np.load(path)
    return d["K"], d["dist"]


def detect_aruco(gray: np.ndarray):
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gray)
    return corners, ids


def build_correspondences(corners, ids, layout):
    obj_list, img_list, used_ids = [], [], []
    for i, mid in enumerate(ids.flatten()):
        if mid not in layout:
            continue
        obj_list.append(layout[mid])
        img_list.append(corners[i].reshape(4, 2))
        used_ids.append(int(mid))
    missing = sorted(set(layout.keys()) - set(used_ids))
    if missing:
        sys.exit(f"마커 {missing} 미검출. 검출된 IDs: {ids.flatten().tolist()}")
    return np.vstack(obj_list).astype(np.float64), np.vstack(img_list).astype(np.float64), used_ids


def visualize(frame, img_pts, proj_pts, mean_err, rvec, tvec, K, dist) -> np.ndarray:
    vis = frame.copy()
    for det, rep in zip(img_pts.astype(int), proj_pts.astype(int)):
        cv2.circle(vis, tuple(det), 5, (0, 255, 0), -1)
        cv2.drawMarker(vis, tuple(rep), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.line(vis, tuple(det), tuple(rep), (255, 165, 0), 1)
    cv2.drawFrameAxes(vis, K, dist, rvec, tvec, length=20, thickness=3)
    R, _ = cv2.Rodrigues(rvec)
    cam  = -R.T @ tvec
    for i, (text, color) in enumerate([
        (f"Mean reproj error: {mean_err:.3f} px", (255, 255, 255)),
        (f"Cam pos (cm)  X={cam[0,0]:.1f}  Y={cam[1,0]:.1f}  Z={cam[2,0]:.1f}", (200, 200, 50)),
    ]):
        cv2.putText(vis, text, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam",   default="52")
    ap.add_argument("--image", default=None)
    args = ap.parse_args()

    layout    = load_layout(LAYOUT_YAML)
    K, dist   = load_intrinsics(args.cam)

    if args.image:
        frame = cv2.imread(args.image) or sys.exit(f"이미지 읽기 실패: {args.image}")
    else:
        frame = grab_frame(RTSP_TEMPLATE.format(cam=args.cam))

    frame = cv2.flip(frame, 1)

    corners, ids = detect_aruco(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        sys.exit("마커 미검출.")
    print(f"검출 IDs: {ids.flatten().tolist()}")

    obj_pts, img_pts, used_ids = build_correspondences(corners, ids, layout)
    print(f"매칭 IDs: {used_ids}  ({len(obj_pts)}쌍)")

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        sys.exit("solvePnP 실패.")

    R, _    = cv2.Rodrigues(rvec)
    cam_pos = -R.T @ tvec
    print(f"rvec: {rvec.flatten()}")
    print(f"tvec: {tvec.flatten()}")
    print(f"카메라 위치 (cm)  X={cam_pos[0,0]:.2f}  Y={cam_pos[1,0]:.2f}  Z={cam_pos[2,0]:.2f}")

    proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj_pts    = proj_pts.reshape(-1, 2)
    errors      = np.linalg.norm(proj_pts - img_pts, axis=1)
    print(f"재투영 오차  mean={errors.mean():.4f}px  max={errors.max():.4f}px")

    cv2.imwrite(str(OUT_DIR / "solvepnp_result.jpg"),
                visualize(frame, img_pts, proj_pts, errors.mean(), rvec, tvec, K, dist))
    np.save(str(OUT_DIR / "rvec.npy"), rvec)
    np.save(str(OUT_DIR / "tvec.npy"), tvec)
    print("저장 완료 → solvepnp_result.jpg, rvec.npy, tvec.npy")


if __name__ == "__main__":
    main()
