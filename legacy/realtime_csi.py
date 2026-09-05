"""
realtime_csi.py
---------------
Real-time lane segmentation using CSI ribbon camera on Jetson Nano.
Uses GStreamer pipeline for CSI camera capture.

Usage:
    python3 realtime_csi.py
    python3 realtime_csi.py --width 640 --height 480 --fps 30
    python3 realtime_csi.py --no-display --save output.mp4
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor

# ── Colour palette (BGR) ─────────────────────────────────────────────────────
CLASS_COLORS_BGR = {
    0: (0,   0,   0),      # background   — black
    1: (200, 200, 200),    # road         — grey
    2: (200,   0, 200),    # divider-line — purple
    3: (0,    0, 255),     # dotted-line  — RED
    4: (0,  140, 255),     # double-line  — orange
    5: (0,  255, 255),     # random-line  — YELLOW
    6: (255,   0,   0),    # road-sign    — BLUE
    7: (0,  255,   0),     # solid-line   — GREEN
}

CLASS_NAMES = {
    0: "background",     1: "road",
    2: "divider-line",   3: "dotted-line",
    4: "double-line",    5: "random-line",
    6: "road-sign-line", 7: "solid-line",
}


def gstreamer_pipeline(width=640, height=480, fps=30, flip=0):
    """Build GStreamer pipeline string for Jetson CSI camera."""
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=1"
    )


def load_model(checkpoint_path, config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    num_labels = cfg["model"]["num_labels"]
    id2label   = {int(k): v for k, v in cfg["model"]["id2label"].items()}
    label2id   = cfg["model"]["label2id"]

    seg_config = SegformerConfig.from_pretrained(
        cfg["model"]["name"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model = SegformerForSemanticSegmentation(seg_config)
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    print(f"✓ Model loaded | Device: {device}")
    return model, cfg, device


def draw_legend(frame, detected):
    h, w     = frame.shape[:2]
    padding  = 10
    box_size = 18
    classes  = [c for c in sorted(detected) if c != 0]
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
        name  = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        cv2.rectangle(frame, (x0 + padding, y),
                      (x0 + padding + box_size, y + box_size), color, -1)
        cv2.putText(frame, name,
                    (x0 + padding + box_size + 6, y + box_size - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += box_size + 6
    return frame


def run(args):
    # Load model
    model, cfg, device = load_model(args.checkpoint, args.config)
    size = cfg["data"]["image_size"]

    processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": size, "width": size},
        do_normalize=True,
        image_mean=cfg["data"]["mean"],
        image_std=cfg["data"]["std"],
    )

    # Open CSI camera
    pipeline = gstreamer_pipeline(
        width=args.width,
        height=args.height,
        fps=args.fps,
        flip=args.flip
    )
    print(f"\nOpening CSI camera...")
    print(f"  Resolution : {args.width}×{args.height} @ {args.fps}fps")
    print(f"  Flip method: {args.flip}")
    print(f"  Pipeline   : {pipeline}\n")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ Failed to open CSI camera!")
        print("   Try: nvgstcapture-1.0 to test camera first")
        print("   Or check flip method (0, 2 are common values)")
        return

    print("✓ CSI camera opened successfully!")
    print("  Press 'q' to quit, 's' to save screenshot\n")

    # Video writer (optional)
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, args.fps,
                                 (args.width, args.height))
        print(f"  Saving to: {args.save}")

    fps_smooth  = 0.0
    frame_count = 0
    skip        = args.skip

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_count += 1
            t0 = time.time()

            # Run inference every `skip` frames
            if frame_count % skip == 0:
                h, w = frame.shape[:2]
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pv   = processor(images=Image.fromarray(rgb),
                                 return_tensors="pt")["pixel_values"].to(device)

                with torch.no_grad():
                    logits    = model(pixel_values=pv).logits
                    upsampled = F.interpolate(logits, size=(h, w),
                                             mode="bilinear", align_corners=False)
                    mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                # Colorize mask
                color_mask = np.zeros_like(frame)
                for cls_id, color in CLASS_COLORS_BGR.items():
                    if cls_id == 0:
                        continue
                    color_mask[mask == cls_id] = color

                vis      = cv2.addWeighted(color_mask, 0.55, frame, 0.45, 0)
                detected = set(np.unique(mask).tolist())
                fps_smooth = 0.9 * fps_smooth + 0.1 / max(time.time() - t0, 1e-6)

                vis = draw_legend(vis, detected)
                cv2.putText(vis, f"FPS: {fps_smooth:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                last_vis = vis

            if writer and 'last_vis' in locals():
                writer.write(last_vis)

            if args.display and 'last_vis' in locals():
                cv2.imshow("Lane Segmentation - CSI Camera", last_vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nStopped by user.")
                    break
                elif key == ord("s"):
                    fname = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(fname, last_vis)
                    print(f"  Screenshot saved: {fname}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Done. Average FPS: {fps_smooth:.1f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/exported/best.pth",
                   help="Path to model checkpoint")
    p.add_argument("--config",     default="segformer_b0.yaml",
                   help="Path to config YAML")
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps",    type=int, default=30)
    p.add_argument("--flip",   type=int, default=0,
                   help="Camera flip: 0=none, 2=rotate180")
    p.add_argument("--skip",   type=int, default=2,
                   help="Run inference every N frames (higher=faster)")
    p.add_argument("--save",   default=None,
                   help="Save output to video file")
    p.add_argument("--no-display", dest="display", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 55)
    print("  Real-time Lane Segmentation — CSI Camera")
    print("=" * 55)
    run(args)
