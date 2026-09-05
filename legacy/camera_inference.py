"""
camera_inference.py
-------------------
Real-time segmentation on Jetson Nano using a TensorRT engine.
Supports CSI camera, USB camera, RTSP stream, and video file.

Usage:
    # CSI camera (default)
    python deployment/jetson/camera_inference.py --engine models/exported/segformer_b0.trt

    # USB camera
    python deployment/jetson/camera_inference.py --engine models/exported/segformer_b0.trt --source usb

    # Video file
    python deployment/jetson/camera_inference.py --engine models/exported/segformer_b0.trt --source file --path video.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent.parent / "tensorrt"))
from infer_trt import SegFormerTRT


# ── Colour palette (BGR) for each class ──────────────────────────────────────
CLASS_COLORS = {
    0: (0,   0,   0),     # background — black
    1: (0, 255,   0),     # road       — green
}


def build_gstreamer_csi(width: int, height: int, fps: int, flip: int) -> str:
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink"
    )


def overlay_mask(frame: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.5) -> np.ndarray:
    """Blend a colour-coded segmentation mask onto the frame."""
    colour_mask = np.zeros_like(frame)
    for cls_id, colour in CLASS_COLORS.items():
        colour_mask[mask == cls_id] = colour

    return cv2.addWeighted(colour_mask, alpha, frame, 1 - alpha, 0)


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return frame


def run(engine_path: str, cfg: dict, source: str, file_path: str = None,
        show: bool = True, save_path: str = None):

    deploy_cfg = cfg["camera"]
    infer_cfg  = cfg["inference"]
    w, h       = infer_cfg["input_size"]

    # ── Open capture ──
    if source == "csi":
        pipeline = build_gstreamer_csi(
            deploy_cfg["width"], deploy_cfg["height"],
            deploy_cfg["fps"], deploy_cfg["flip_method"]
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif source == "usb":
        cap = cv2.VideoCapture(deploy_cfg.get("usb_device", "/dev/video0"))
    elif source == "rtsp":
        cap = cv2.VideoCapture(deploy_cfg["rtsp_url"])
    elif source == "file":
        cap = cv2.VideoCapture(file_path)
    else:
        raise ValueError(f"Unknown source: {source}")

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/stream: {source}")

    # ── Load TRT engine ──
    print(f"Loading TensorRT engine: {engine_path}")
    model = SegFormerTRT(engine_path)
    print("Engine loaded. Starting inference...\n")

    # ── Video writer ──
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, deploy_cfg["fps"], (w, h))

    # ── Inference loop ──
    fps_smooth = 0.0
    prev_time  = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream.")
                break

            # Resize to model input
            resized = cv2.resize(frame, (w, h))

            # Predict
            t0   = time.time()
            mask = model.predict_mask(resized)
            t1   = time.time()

            # FPS
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(t1 - t0, 1e-6))

            # Overlay
            vis = overlay_mask(resized, mask, alpha=infer_cfg["overlay_alpha"])
            if cfg["display"]["show_fps"]:
                vis = draw_fps(vis, fps_smooth)

            if writer:
                writer.write(vis)

            if show:
                cv2.imshow("Dotted-Line Segmentation", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\nDone. Average FPS: {fps_smooth:.1f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="models/exported/segformer_b0.trt")
    p.add_argument("--config", default="configs/jetson_deploy.yaml")
    p.add_argument("--source", default="csi",
                   choices=["csi", "usb", "rtsp", "file"])
    p.add_argument("--path",   default=None,
                   help="Video file path (only for --source file)")
    p.add_argument("--no-show", dest="show", action="store_false")
    p.add_argument("--save",   default=None,
                   help="Path to save output video (e.g. output.mp4)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(
        engine_path=args.engine,
        cfg=cfg,
        source=args.source,
        file_path=args.path,
        show=args.show,
        save_path=args.save,
    )
