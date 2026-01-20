#!/usr/bin/env python3
"""
Warp 2D points from multiple source images onto a target image (ground-plane homography).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def load_refined_target(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Refined target camera not found: {path}")
    with path.open("r") as f:
        data = json.load(f)
    if "K" not in data or "R_w2c" not in data or "t_w2c" not in data:
        raise RuntimeError("Refined target JSON missing K/R_w2c/t_w2c")
    return {
        "target_image": data.get("target_image"),
        "K": np.array(data["K"], dtype=np.float64),
        "R_w2c": np.array(data["R_w2c"], dtype=np.float64),
        "t_w2c": np.array(data["t_w2c"], dtype=np.float64),
    }


def load_refined_params(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Refined camera params not found: {path}")
    with path.open("r") as f:
        data = json.load(f)
    return data.get("optimized", {})


def load_cameras_txt(path: Path):
    cameras = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(p) for p in parts[4:]], dtype=np.float64)
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def load_images_txt(path: Path):
    images = {}
    with path.open("r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split(maxsplit=9)
        if len(parts) < 10:
            i += 1
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        images[name] = {
            "image_id": image_id,
            "qw": qw,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "tx": tx,
            "ty": ty,
            "tz": tz,
            "camera_id": camera_id,
        }
        i += 2
    return images


def intrinsic_from_model(camera, target_shape=None):
    model = camera["model"]
    params = camera["params"]
    width = camera["width"]
    height = camera["height"]

    if model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = params[:4]
    elif model in ("RADIAL", "SIMPLE_RADIAL"):
        f = params[0]
        cx = params[1]
        cy = params[2]
        fx = f
        fy = f
    else:
        if len(params) >= 4:
            fx, fy, cx, cy = params[:4]
        elif len(params) >= 3:
            f, cx, cy = params[:3]
            fx = f
            fy = f
        else:
            raise ValueError(f"Unsupported camera model: {model}")

    if target_shape is not None:
        img_h, img_w = target_shape
        if img_w != width or img_h != height:
            sx = img_w / float(width)
            sy = img_h / float(height)
            fx *= sx
            fy *= sy
            cx *= sx
            cy *= sy

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K


def points_xy(points):
    if not points:
        return []
    if isinstance(points[0], dict):
        return [[float(p["x"]), float(p["y"])] for p in points]
    return [[float(p[0]), float(p[1])] for p in points]


def warp_points(points, H):
    pts = np.array(points_xy(points), dtype=np.float64)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    warped = (H @ pts_h.T).T
    warped = warped[:, :2] / warped[:, 2:3]
    return warped


def draw_points(canvas, points, radius, color=(0, 255, 0)):
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < canvas.shape[1] and 0 <= yi < canvas.shape[0]:
            cv2.circle(canvas, (xi, yi), radius, color, thickness=-1, lineType=cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSON mapping image -> points")
    parser.add_argument("--refined", type=Path, default=Path("outputs/refined_camera_params.json"))
    parser.add_argument("--refined-target", type=Path, default=Path("outputs/refined_target_camera.json"))
    parser.add_argument("--poses", type=Path, default=Path("outputs/transformed_images.txt"))
    parser.add_argument("--cameras", type=Path, default=Path("sparse/0/txt/cameras.txt"))
    parser.add_argument("--target-image", type=str, default="target.png")
    parser.add_argument("--target-image-path", type=Path, default=Path("images/target"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/warped_points_target.json"))
    parser.add_argument("--output-image", type=Path, default=Path("outputs/annotated/target_points.png"))
    parser.add_argument("--radius", type=int, default=5)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input}")
    if not args.refined.exists():
        raise FileNotFoundError(f"Refined camera params not found: {args.refined}")

    target_path = args.target_image_path / args.target_image
    if not target_path.exists():
        raise FileNotFoundError(f"Target image not found: {target_path}")

    target_img = cv2.imread(str(target_path))
    if target_img is None:
        raise RuntimeError(f"Could not read target image: {target_path}")

    with args.input.open("r") as f:
        input_points = json.load(f)
    refined_map = load_refined_params(args.refined)

    images = {}
    cameras = {}
    if args.poses.exists() and args.cameras.exists():
        images = load_images_txt(args.poses)
        cameras = load_cameras_txt(args.cameras)

    target_h, target_w = target_img.shape[:2]
    if args.target_image in images and cameras:
        tgt = images[args.target_image]
        cam_id = tgt["camera_id"]
        if cam_id not in cameras:
            raise RuntimeError(f"Target camera_id {cam_id} not in cameras.txt")
        K = intrinsic_from_model(cameras[cam_id], target_shape=(target_h, target_w))
        qx, qy, qz, qw = tgt["qx"], tgt["qy"], tgt["qz"], tgt["qw"]
        R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        t_w2c = np.array([tgt["tx"], tgt["ty"], tgt["tz"]], dtype=np.float64)
        print("Using target pose from transformed_images.txt")
    else:
        refined_target = load_refined_target(args.refined_target)
        if refined_target["target_image"] and refined_target["target_image"] != args.target_image:
            raise RuntimeError(
                f"Refined target camera is for {refined_target['target_image']}, "
                f"not {args.target_image}"
            )
        K = refined_target["K"]
        R_w2c = refined_target["R_w2c"]
        t_w2c = refined_target["t_w2c"]
        print("Using refined target camera")

    H_w2img = K @ np.hstack([R_w2c[:, :2], t_w2c.reshape(3, 1)])

    warped_out = {"target_image": args.target_image, "points": {}}

    canvas = target_img.copy()
    for img_name, pts in input_points.items():
        if img_name not in refined_map:
            print(f"WARNING: {img_name} not in refined params; skipping")
            continue
        K_src = np.array(refined_map[img_name]["K"], dtype=np.float64)
        R_src = np.array(refined_map[img_name]["R_w2c"], dtype=np.float64)
        t_src = np.array(refined_map[img_name]["t_w2c"], dtype=np.float64)
        H_src_w2img = K_src @ np.hstack([R_src[:, :2], t_src.reshape(3, 1)])
        H_img2world = np.linalg.inv(H_src_w2img)
        H_img2target = H_w2img @ H_img2world

        warped = warp_points(pts, H_img2target)
        warped_pts = []
        for idx, (x, y) in enumerate(warped):
            item = {"x": float(x), "y": float(y)}
            if isinstance(pts, list) and pts and isinstance(pts[0], dict) and "id" in pts[0]:
                item["id"] = int(pts[idx]["id"])
            warped_pts.append(item)
        warped_out["points"][img_name] = warped_pts

        draw_points(canvas, warped, args.radius)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(warped_out, indent=2) + "\n")

    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_image), canvas)

    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.output_image}")


if __name__ == "__main__":
    main()
