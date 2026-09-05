"""
gst_trt_live_inference.py

Jetson CSI live lane segmentation using TensorRT FP16 engine
instead of PyTorch — significantly faster inference on Jetson GPU.

Usage:
    python3 gst_trt_live_inference.py --engine models/exported/segformer_b0_fp16.engine
    python3 gst_trt_live_inference.py --engine models/exported/segformer_b0_fp16.engine --skip 1
    python3 gst_trt_live_inference.py --engine models/exported/segformer_b0_fp16.engine --no-show --output results/trt_test.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 — initialises CUDA context
import tensorrt as trt


# ── Class colours and names ───────────────────────────────────────────────────

CLASS_COLORS_BGR = {
    0: (0,   0,   0),      # background
    1: (200, 200, 200),    # road
    2: (200, 0,   200),    # divider-line
    3: (0,   0,   255),    # dotted-line
    4: (0,   140, 255),    # double-line
    5: (0,   255, 255),    # random-line
    6: (255, 0,   0),      # road-sign-line
    7: (0,   255, 0),      # solid-line
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

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ── TensorRT engine ───────────────────────────────────────────────────────────

class SegFormerTRT:
    def __init__(self, engine_path: str):
        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self._allocate_buffers()
        print(f"✓ TensorRT engine loaded")

    def _allocate_buffers(self):
        self.inputs  = []
        self.outputs = []

        # TensorRT 10.x replaced the old binding-index API with a tensor-name
        # based API. Iterate I/O tensors by name instead of by binding index.
        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size  = trt.volume(shape)
            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            entry = {"name": name, "host": host_mem, "device": device_mem, "shape": shape}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

            # Bind the device pointer to this tensor name up front. Address
            # doesn't change between calls since we reuse the same buffers.
            self.context.set_tensor_address(name, int(device_mem))

        self.stream = cuda.Stream()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """BGR uint8 image → normalised float32 tensor (1×3×640×640)"""
        img = cv2.resize(image, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        return img.transpose(2, 0, 1)[np.newaxis]  # 1×3×640×640

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        """Run TensorRT inference, return segmentation mask (H×W uint8)"""
        h, w   = image.shape[:2]
        tensor = self.preprocess(image)
        np.copyto(self.inputs[0]["host"], tensor.ravel())

        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()

        logits = self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        mask   = logits[0].argmax(axis=0).astype(np.uint8)
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


# ── GStreamer camera ──────────────────────────────────────────────────────────

def build_pipeline(args):
    return (
        f"nvarguscamerasrc sensor-id={args.sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={args.capture_width}, height={args.capture_height}, "
        f"format=NV12, framerate={args.fps}/1 ! "
        f"nvvidconv flip-method={args.flip_method} ! "
        f"video/x-raw, format=BGRx, width={args.display_width}, height={args.display_height} ! "
        "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
    )


class GstCamera:
    def __init__(self, pipeline_str):
        Gst.init(None)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink  = self.pipeline.get_by_name("sink")
        if self.appsink is None:
            raise RuntimeError("Could not find appsink in GStreamer pipeline.")
        self.bus = self.pipeline.get_bus()

    def start(self):
        result = self.pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start GStreamer camera pipeline.")

        # Block until the pipeline actually reaches PLAYING (or errors out),
        # instead of returning immediately and racing the appsink read.
        state_result, state, pending = self.pipeline.get_state(10 * Gst.SECOND)
        if state_result == Gst.StateChangeReturn.FAILURE:
            self._check_bus_errors(raise_on_error=True)
            raise RuntimeError("Pipeline failed to reach PLAYING state (no bus error captured).")
        print(f"Pipeline state: {state.value_nick} (result={state_result.value_nick})")

    def _check_bus_errors(self, raise_on_error=False):
        while True:
            msg = self.bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.WARNING)
            if msg is None:
                break
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"GST ERROR from {msg.src.get_name()}: {err} | debug: {debug}")
                if raise_on_error:
                    raise RuntimeError(f"GStreamer pipeline error: {err}")
            elif msg.type == Gst.MessageType.WARNING:
                warn, debug = msg.parse_warning()
                print(f"GST WARNING from {msg.src.get_name()}: {warn} | debug: {debug}")

    def read(self, timeout_ns=2_000_000_000):
        sample = self.appsink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            self._check_bus_errors()
            return False, None

        buffer    = sample.get_buffer()
        caps      = sample.get_caps()
        structure = caps.get_structure(0)
        width     = structure.get_value("width")
        height    = structure.get_value("height")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return False, None

        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 4)).copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        finally:
            buffer.unmap(map_info)

        return True, frame

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)


# ── Visualisation helpers ─────────────────────────────────────────────────────

def draw_legend(frame, detected):
    h, w      = frame.shape[:2]
    padding   = 10
    box_size  = 18
    classes   = [c for c in sorted(detected) if c != 0]
    if not classes:
        return frame

    legend_h  = padding + len(classes) * (box_size + 6) + padding
    legend_w  = 180
    x0        = w - legend_w - padding
    y0        = padding

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + legend_w, y0 + legend_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = y0 + padding
    for cls_id in classes:
        color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))
        name  = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        cv2.rectangle(frame, (x0 + padding, y), (x0 + padding + box_size, y + box_size), color, -1)
        cv2.putText(frame, name,
                    (x0 + padding + box_size + 6, y + box_size - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += box_size + 6
    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    model        = SegFormerTRT(args.engine)
    pipeline_str = build_pipeline(args)

    print("Opening CSI camera with pipeline:")
    print(pipeline_str)

    camera = GstCamera(pipeline_str)
    camera.start()

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
    ok, frame = camera.read()
    if not ok or frame is None:
        camera.stop()
        if writer:
            writer.release()
        raise RuntimeError("Camera pipeline started but no frame received from appsink.")

    print(f"✓ First frame received: {frame.shape[1]}x{frame.shape[0]}")
    print("✓ Running inference with TensorRT FP16 engine\n")

    alpha       = 0.55
    fps_smooth  = 0.0
    frame_count = 0
    last_vis    = None

    try:
        while True:
            if frame_count > 0:
                ok, frame = camera.read()
                if not ok or frame is None:
                    print("No frame received; stopping.")
                    break

            h, w = frame.shape[:2]
            frame_count += 1

            if frame_count % args.skip == 0 or last_vis is None:
                t0   = time.time()
                mask = model.predict_mask(frame)

                color_mask = np.zeros_like(frame)
                for cls_id, color in CLASS_COLORS_BGR.items():
                    if cls_id == 0:
                        continue
                    color_mask[mask == cls_id] = color

                vis      = cv2.addWeighted(color_mask, alpha, frame, 1 - alpha, 0)
                detected = set(np.unique(mask).tolist())
                elapsed  = max(time.time() - t0, 1e-6)
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / elapsed)

                vis = draw_legend(vis, detected)
                cv2.putText(vis, f"TRT FPS: {fps_smooth:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2, cv2.LINE_AA)
                last_vis = vis

            if writer:
                writer.write(last_vis)

            if args.show:
                cv2.imshow("TensorRT Lane Segmentation", last_vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break

            if frame_count % 30 == 0:
                print(f"Frames: {frame_count} | TRT FPS: {fps_smooth:.1f}")

    finally:
        camera.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print("✓ Inference stopped cleanly")
    if args.output:
        print(f"✓ Saved output to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson CSI TensorRT lane segmentation")
    parser.add_argument("--engine",         required=True, help="Path to TensorRT .engine file")
    parser.add_argument("--sensor-id",      type=int, default=0)
    parser.add_argument("--capture-width",  type=int, default=1280)
    parser.add_argument("--capture-height", type=int, default=720)
    parser.add_argument("--display-width",  type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument("--fps",            type=int, default=30)
    parser.add_argument("--flip-method",    type=int, default=0)
    parser.add_argument("--skip",           type=int, default=1)
    parser.add_argument("--output",         default=None)
    parser.add_argument("--no-show",        dest="show", action="store_false")
    parser.set_defaults(show=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
