import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

RTSP_TEMPLATE = os.environ["RTSP_TEMPLATE"]
LAYOUT_YAML   = Path(__file__).parent.parent / "marker_layout.yaml"
ARUCO_DICT    = cv2.aruco.DICT_4X4_100
OUT_DIR       = Path(__file__).parent


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
    if not obj_list:
        sys.exit(f"레이아웃 매칭 실패. 검출: {ids.flatten().tolist()}  레이아웃: {sorted(layout)}")
    return np.vstack(obj_list), np.vstack(img_list), used_ids


def dlt(obj_pts: np.ndarray, img_pts: np.ndarray) -> np.ndarray:
    """Normalised DLT (Hartley & Zisserman §4.1)"""
    n = len(obj_pts)
    if n < 6:
        sys.exit(f"DLT 최소 6쌍 필요 (현재 {n}쌍).")

    mu3, s3 = obj_pts.mean(0), np.sqrt(((obj_pts - obj_pts.mean(0)) ** 2).sum(1)).mean() or 1.0
    T = np.diag([1/s3, 1/s3, 1/s3, 1.0])
    T[:3, 3] = -mu3 / s3

    mu2, s2 = img_pts.mean(0), np.sqrt(((img_pts - img_pts.mean(0)) ** 2).sum(1)).mean() or 1.0
    U = np.array([[1/s2, 0, -mu2[0]/s2], [0, 1/s2, -mu2[1]/s2], [0, 0, 1]])

    A = np.zeros((2 * n, 12))
    for i in range(n):
        X    = (T @ np.append(obj_pts[i], 1.0))[:3]
        u, v = (U @ np.append(img_pts[i], 1.0))[:2]
        A[2*i]   = [*X, 1,  0,  0,  0,  0, -u*X[0], -u*X[1], -u*X[2], -u]
        A[2*i+1] = [ 0,  0,  0,  0, *X, 1, -v*X[0], -v*X[1], -v*X[2], -v]

    P = np.linalg.inv(U) @ np.linalg.svd(A)[2][-1].reshape(3, 4) @ T
    return P if P[2, 3] > 0 else -P


def project(P: np.ndarray, pts3d: np.ndarray) -> np.ndarray:
    h = np.hstack([pts3d, np.ones((len(pts3d), 1))]) @ P.T
    return h[:, :2] / h[:, 2:3]


def visualize(frame, img_pts, proj_pts, mean_err) -> np.ndarray:
    vis = frame.copy()
    for det, rep in zip(img_pts.astype(int), proj_pts.astype(int)):
        cv2.circle(vis, tuple(det), 5, (0, 255, 0), -1)
        cv2.drawMarker(vis, tuple(rep), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.line(vis, tuple(det), tuple(rep), (255, 165, 0), 1)
    cv2.putText(vis, f"Mean reprojection error: {mean_err:.3f} px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam",   default="52")
    ap.add_argument("--image", default=None)
    args = ap.parse_args()

    layout = load_layout(LAYOUT_YAML)

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

    P      = dlt(obj_pts.astype(np.float64), img_pts.astype(np.float64))
    reproj = project(P, obj_pts)
    errors = np.linalg.norm(reproj - img_pts, axis=1)
    print(f"P:\n{np.array2string(P, precision=4, suppress_small=True)}")
    print(f"재투영 오차  mean={errors.mean():.4f}px  max={errors.max():.4f}px")

    cv2.imwrite(str(OUT_DIR / "dlt_result.jpg"), visualize(frame, img_pts, reproj, errors.mean()))
    np.save(str(OUT_DIR / "projection_matrix.npy"), P)
    print("저장 완료 → dlt_result.jpg, projection_matrix.npy")


if __name__ == "__main__":
    main()
