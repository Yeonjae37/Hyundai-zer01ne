import cv2
import numpy as np
import glob
import os
import sys

BOARD_COLS  = 7     # inner corners horizontally
BOARD_ROWS  = 10    # inner corners vertically
SQUARE_MM   = 30    # physical square size in mm
MIN_FRAMES  = 10
TARGET_SIZE = (2560, 1440)   # (width, height)


def find_corners(gray, board_size):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if found:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return found, corners


def calibrate(cam_num: str):
    board_size = (BOARD_COLS, BOARD_ROWS)
    frame_dir  = os.path.join("frames", cam_num)
    out_dir    = os.path.join("params", cam_num)

    images = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not images:
        print(f"No images found in {frame_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images  |  target: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_MM

    obj_pts, img_pts = [], []
    skipped = 0

    for path in images:
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        if gray.shape[::-1] != TARGET_SIZE:
            skipped += 1
            continue
        found, corners = find_corners(gray, board_size)
        if found:
            obj_pts.append(objp)
            img_pts.append(corners)

    if skipped:
        print(f"Skipped {skipped} images (resolution mismatch)")
    print(f"Valid frames: {len(obj_pts)} / {len(images) - skipped}")

    if len(obj_pts) < MIN_FRAMES:
        print(f"Not enough valid frames (need >= {MIN_FRAMES})")
        sys.exit(1)

    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, TARGET_SIZE, None, None
    )

    print(f"\nRMS error : {rms:.4f} px")
    print(f"K :\n{K}")
    print(f"dist: {dist.ravel()}")

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "intrinsics.npz"),
             K=K, dist=dist, rms=rms, img_size=np.array(TARGET_SIZE))

    fs = cv2.FileStorage(os.path.join(out_dir, "intrinsics.yaml"), cv2.FILE_STORAGE_WRITE)
    fs.write("K", K)
    fs.write("dist", dist)
    fs.write("rms", rms)
    fs.write("img_width",  TARGET_SIZE[0])
    fs.write("img_height", TARGET_SIZE[1])
    fs.release()

    print(f"\nSaved -> {out_dir}/intrinsics.{{npz,yaml}}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {os.path.basename(__file__)} <cam_num>")
        sys.exit(1)
    calibrate(sys.argv[1])
