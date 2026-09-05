"""
predict_image.py
----------------
Run SegFormer on a single image — original vs segmented side by side.

Usage:
    python scripts/predict_image.py --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerFeatureExtractor

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
        id2label=id2label,
	label2id=label2id,
    )
    model = SegformerForSemanticSegmentation(seg_config)
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    print(f"✓ Model loaded from {checkpoint_path}")
    return model


def draw_legend(vis, detected):
    y = 20
    for cls_id in sorted(detected):
        if cls_id == 0:
            continue
        color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))
        name  = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        cv2.rectangle(vis, (10, y), (28, y + 18), color, -1)
        cv2.putText(vis, name, (34, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return vis


def add_title(img, text):
    bar = np.zeros((40, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def predict(image_path, checkpoint, config, output, alpha, display):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    size = int(cfg["data"]["size"])
    processor = SegformerFeatureExtractor(
        do_resize=True, size=size,
        do_normalize=True,
        image_mean=cfg["data"]["mean"],
        image_std=cfg["data"]["std"],
    )
    model = load_model(checkpoint, cfg)

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")
    orig_h, orig_w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pv  = processor(Image.fromarray(rgb), return_tensors="pt")["pixel_values"]

    with torch.no_grad():
        logits    = model(pixel_values=pv).logits
        upsampled = F.interpolate(logits, size=(orig_h, orig_w),
                                  mode="bilinear", align_corners=False)
        mask = upsampled.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)

    # Colorize
    color_mask = np.zeros_like(frame)
    for cls_id, color in CLASS_COLORS_BGR.items():
        if cls_id == 0:
            continue
        color_mask[mask == cls_id] = color

    segmented = cv2.addWeighted(color_mask, alpha, frame, 1 - alpha, 0)
    detected  = [c for c in np.unique(mask).tolist() if c != 0]

    print(f"\nDetected: {[CLASS_NAMES[c] for c in detected]}")

    segmented = draw_legend(segmented, detected)

    # Side by side
    left  = add_title(frame.copy(), "Original")
    right = add_title(segmented,    "Segmented")
    div   = np.full((left.shape[0], 3, 3), 60, dtype=np.uint8)
    combined = np.hstack([left, div, right])

    # Resize if too wide
    if combined.shape[1] > 1400:
        scale    = 1400 / combined.shape[1]
        combined = cv2.resize(combined,
                              (int(combined.shape[1]*scale), int(combined.shape[0]*scale)))

    if output is None:
        stem   = Path(image_path).stem
        output = f"results/predictions/{stem}_segmented.jpg"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output, combined)
    print(f"✓ Saved → {output}")

    if display:
        cv2.imshow("Original  |  Segmented", combined)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",      required=True)
    p.add_argument("--checkpoint", default="models/checkpoints/best.pth")
    p.add_argument("--config",     default="configs/segformer_b0.yaml")
    p.add_argument("--output",     default=None)
    p.add_argument("--alpha",      type=float, default=0.6)
    p.add_argument("--no-display", dest="display", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 55)
    print("  Lane Segmentation — Image Inference")
    print("=" * 55)
    predict(args.image, args.checkpoint, args.config,
            args.output, args.alpha, args.display)
