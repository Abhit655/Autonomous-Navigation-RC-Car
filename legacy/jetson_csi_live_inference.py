"""
jetson_csi_live_inference.py

Lane segmentation live inference for Jetson CSI ribbon cameras using GStreamer.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import (
    SegformerConfig,
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
)


CLASS_COLORS_BGR = {
    0: (0, 0, 0),
    1: (200, 200, 200),
    2: (200, 0, 200),
    3: (0, 0, 255),
    4: (0, 140, 255),
    5: (0, 255, 255),
    6: (255, 0, 0),
    7: (0, 255, 0),
}

CLASS_NAMES = {
    0: "background",
    1: "road",
    2: "divider-line",
    3: "dotted-line",
    4: "double-line",
    5: "random-line",
    6: "road-sign-line",
    7: "solid-line",
}


def load_model(checkpoint_path, cfg, device):
    num_labels = cfg["model"]["num_labels"]
    id2label = {int(k): v for k, v in cfg["model"]["id2label"].items()}
    label2id = cfg["model"]["label2id"]

    seg_config = SegformerConfig(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    model = SegformerForSemanticSegmentation(seg_config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    print(f"✓ Model loaded from {checkpoint_path}")
    print(f"✓ Running on device: {device}")
    return model


def fix_rotation(frame, rotation):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def draw_legend(frame, detected):
    h, w = frame.shape[:2]
    padding = 10
    box_size = 18
    classes = [c for c in sorted(detected) if c != 0]
    if not classes:
        return frame

    legend_h = padding + len(classes) * (box_size + 6) + padding
    legend_w = 180
    x0 = w - legend_w - padding
    y0 = padding

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + legend_w, y0 + legend_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = y0 + padding
    for cls_id in classes:
        color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        cv2.rectangle(
            frame,
            (x0 + padding, y),
            (x0 + padding + box_size, y + box_size),
            color,
            -1,
        )
        cv2.putText(
            frame,
            name,
            (x0 + padding + box_size + 6, y + box_size - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += box_size + 6
    return frame


def build_csi_pipeline(sensor_id, capture_width, capture_height, display_width, display_height, fps, flip_method):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def open_csi_camera(args):
    pipeline = build_csi_pipeline(
        sensor_id=args.sensor_id,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        display_width=args.display_width,
        display_height=args.display_height,
        fps=args.fps,
        flip_method=args.flip_method,
    )
    print("Opening CSI camera with pipeline:")
    print(pipeline)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            "Failed to open Jetson CSI camera via GStreamer. "
            "Check that the ribbon camera is connected, enabled, and not in use."
        )
    return cap


def process_live(args, model, processor, device):
    cap = open_csi_camera(args)

    writer = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.display_width, args.display_height),
        )

    print("Reading first frame...")
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        if writer is not None:
            writer.release()
        raise RuntimeError(
            "Camera opened but no frame was returned. "
            "This usually means the CSI camera pipeline is not producing frames."
        )
    print(f"✓ First frame received: {frame.shape[1]}x{frame.shape[0]}")

    alpha = 0.55
    fps_smooth = 0.0
    frame_count = 0
    last_vis = None

    try:
        while True:
            if frame_count > 0:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("No more frames from camera; stopping.")
                    break

            frame = fix_rotation(frame, args.rotate)
            h, w = frame.shape[:2]
            frame_count += 1

            if frame_count % args.skip == 0 or last_vis is None:
                t0 = time.time()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pixel_values = processor(Image.fromarray(rgb), return_tensors="pt")["pixel_values"].to(device)

                with torch.no_grad():
                    logits = model(pixel_values=pixel_values).logits
                    upsampled = F.interpolate(
                        logits,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    mask = upsampled.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

                color_mask = np.zeros_like(frame)
                for cls_id, color in CLASS_COLORS_BGR.items():
                    if cls_id == 0:
                        continue
                    color_mask[mask == cls_id] = color

                vis = cv2.addWeighted(color_mask, alpha, frame, 1 - alpha, 0)
                detected = set(np.unique(mask).tolist())
                elapsed = max(time.time() - t0, 1e-6)
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / elapsed)
                vis = draw_legend(vis, detected)
                cv2.putText(
                    vis,
                    f"FPS: {fps_smooth:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                last_vis = vis

            if args.show:
                show = last_vis
                dh, dw = show.shape[:2]
                if dh > 900:
                    scale = 900 / dh
                    show = cv2.resize(show, (int(dw * scale), 900))
                cv2.imshow("Jetson CSI Lane Segmentation", show)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Stopped by user.")
                    break

            if writer is not None:
                writer.write(last_vis)

            if frame_count % 30 == 0:
                print(f"Frames processed: {frame_count} | FPS: {fps_smooth:.1f}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print("✓ Live inference stopped cleanly")
    if args.output:
        print(f"✓ Saved output video to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson CSI live lane segmentation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/segformer_b0.yaml")
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--capture-width", type=int, default=1280)
    parser.add_argument("--capture-height", type=int, default=720)
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip-method", type=int, default=0)
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    size = cfg["data"]["size"]
    processor = SegformerFeatureExtractor(
        do_resize=True,
        size=(size, size),
        do_normalize=True,
        image_mean=cfg["data"]["mean"],
        image_std=cfg["data"]["std"],
    )

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, cfg, device)
    process_live(args, model, processor, device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
