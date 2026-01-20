# Warp Points to Target Image

This package lets you warp 2D points from the five source images onto `target.png` using refined camera parameters.

## Contents
- `scripts/18_warp_points_to_target.py`
- `outputs/refined_camera_params.json`
- `outputs/refined_target_camera.json`
- `outputs/transformed_images.txt` (used if the target image exists in poses)
- `sparse/0/txt/cameras.txt`
- `images/target/target.png`

## Requirements
- Python 3.8+
- `opencv-python`
- `numpy`
- `scipy`

## Input format
Provide a JSON mapping image names to point lists:

```json
{
  "Copy of cam1.png": [{"id": 1, "x": 123.4, "y": 567.8}],
  "Copy of cam2.png": [[100.0, 200.0], [300.0, 400.0]]
}
```

## Usage
From the package root:

```bash
python scripts/18_warp_points_to_target.py \
  --input /path/to/points.json \
  --refined outputs/refined_camera_params.json \
  --refined-target outputs/refined_target_camera.json \
  --target-image target.png \
  --target-image-path images/target \
  --poses outputs/transformed_images.txt \
  --cameras sparse/0/txt/cameras.txt
```

Outputs:
- `outputs/warped_points_target.json`
- `outputs/annotated/target_points.png`

Notes:
- If `target.png` exists in `outputs/transformed_images.txt`, the script uses that pose; otherwise it uses `outputs/refined_target_camera.json`.
- Source image names in your input JSON must match the keys in `outputs/refined_camera_params.json`.
