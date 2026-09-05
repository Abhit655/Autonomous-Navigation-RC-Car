"""
video_inference.py — Lane Segmentation Video Inference
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
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerFeatureExtractor
from tqdm import tqdm

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


def load_model(checkpoint_path, cfg):
    num_labels = cfg["model"]["num_labels"]
    id2label   = {int(k): v for k, v in cfg["model"]["id2label"].items()}
    label2id   = cfg["model"]["label2id"]
    seg_config = SegformerConfig(
        num_labels=num_labels,
        id2label=id2label, label2id=label2id,
    )
    model = SegformerForSemanticSegmentation(seg_config)
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    print(f"✓ Model loaded from {checkpoint_path}")
    return model


def fix_rotation(frame, rotation):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


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


def process_video(video_path, output_path, model, processor, cfg,
                  display=True, rotate=0, skip=1):

    if video_path.startswith("csi://"):
        gst_pipeline = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    alpha        = 0.55

    # Auto-detect rotation
    auto_rot = 0
    rotation = rotate if rotate != 0 else auto_rot
    print(f"  Rotation: {rotation}°")

    if rotation in [90, 270]:
        out_w, out_h = orig_h, orig_w
    else:
        out_w, out_h = orig_w, orig_h

    print(f"  Input : {orig_w}×{orig_h}  →  Output: {out_w}×{out_h}")
    print(f"  FPS: {fps_in:.1f}  |  Frames: {total_frames}  |  Skip: every {skip} frame(s)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_in, (out_w, out_h))

    fps_smooth  = 0.0
    last_vis    = None   # reuse last frame when skipping
    frame_count = 0
    pbar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = fix_rotation(frame, rotation)
        h, w  = frame.shape[:2]
        frame_count += 1

        # Only run inference every `skip` frames
        if frame_count % skip == 0 or last_vis is None:
            t0  = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pv  = processor(Image.fromarray(rgb), return_tensors="pt")["pixel_values"]

            with torch.no_grad():
                logits    = model(pixel_values=pv).logits
                upsampled = F.interpolate(logits, size=(h, w),
                                          mode="bilinear", align_corners=False)
                mask = upsampled.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)

            color_mask = np.zeros_like(frame)
            for cls_id, color in CLASS_COLORS_BGR.items():
                if cls_id == 0:
                    continue
                color_mask[mask == cls_id] = color

            vis      = cv2.addWeighted(color_mask, alpha, frame, 1 - alpha, 0)
            detected = set(np.unique(mask).tolist())
            fps_smooth = 0.9 * fps_smooth + 0.1 / max(time.time() - t0, 1e-6)
            vis = draw_legend(vis, detected)
            cv2.putText(vis, f"FPS: {fps_smooth:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            last_vis = vis
        else:
            # Reuse last prediction on skipped frames
            last_vis = cv2.addWeighted(
                np.zeros_like(frame), 0, frame, 1, 0
            ) if last_vis is None else last_vis

        writer.write(last_vis)

        if display:
            dh, dw = last_vis.shape[:2]
            show   = last_vis
            if dh > 800:
                scale = 800 / dh
                show  = cv2.resize(last_vis, (int(dw * scale), 800))
            cv2.imshow("Lane Segmentation", show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user.")
                break

        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Saved → {output_path}")
    print(f"  Average inference FPS: {fps_smooth:.1f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",      required=True)
    p.add_argument("--checkpoint", default="models/checkpoints/best.pth")
    p.add_argument("--config",     default="configs/segformer_b0.yaml")
    p.add_argument("--output",     default=None)
    p.add_argument("--rotate",     type=int, default=0, choices=[0, 90, 180, 270])
    p.add_argument("--skip",       type=int, default=1,
                   help="Run inference every N frames (2=2x faster, 3=3x faster)")
    p.add_argument("--no-display", dest="display", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output is None:
        stem = Path(args.video).stem
        args.output = f"results/predictions/{stem}_segmented.mp4"

    print("=" * 55)
    print("  Lane Segmentation — Video Inference")
    print("=" * 55)

    size=cfg["data"]["size"]

    processor = SegformerFeatureExtractor(
        do_resize=True,
        size=(size,size),
	do_normalize=True,
        image_mean=cfg["data"]["mean"],
        image_std=cfg["data"]["std"],
    )
    model = load_model(args.checkpoint, cfg)
    process_video(args.video, args.output, model, processor, cfg,
                  display=args.display, rotate=args.rotate, skip=args.skip)
