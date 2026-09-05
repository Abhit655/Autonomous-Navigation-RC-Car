"""
infer_trt.py
------------
TensorRT inference wrapper for SegFormer segmentation engine.
Used by camera_inference.py — can also be imported directly.
"""

from pathlib import Path

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit          # noqa: F401  initialises CUDA context
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class SegFormerTRT:
    """Load a TensorRT engine and run segmentation inference."""

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.engine      = self._load_engine()
        self.context     = self.engine.create_execution_context()
        self._allocate_buffers()

    # ── Setup ──────────────────────────────────────────────────────────────

    def _load_engine(self):
        with open(self.engine_path, "rb") as f, \
             trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        self.inputs  = []
        self.outputs = []

        # Resolve dynamic dims (e.g. batch=-1) to a concrete shape (batch=1)
        # so output shapes can be queried from the context below.
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = self.engine.get_tensor_shape(name)
                shape = tuple(1 if d == -1 else d for d in shape)
                self.context.set_input_shape(name, shape)

        for i in range(self.engine.num_io_tensors):
            name    = self.engine.get_tensor_name(i)
            shape   = self.context.get_tensor_shape(name)
            dtype   = trt.nptype(self.engine.get_tensor_dtype(name))
            size    = trt.volume(shape)
            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.context.set_tensor_address(name, int(device_mem))

            entry = {"name": name, "host": host_mem, "device": device_mem, "shape": shape}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

        self.stream = cuda.Stream()

    # ── Inference ──────────────────────────────────────────────────────────

    def preprocess(self, image: np.ndarray, input_size: tuple = (640, 640)) -> np.ndarray:
        """BGR image (H×W×3 uint8) → normalised float32 tensor (1×3×H×W)."""
        import cv2
        img = cv2.resize(image, input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        return img.transpose(2, 0, 1)[np.newaxis]       # 1×3×H×W

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference on a BGR image.
        Returns logits array shaped (1, num_classes, H', W').
        """
        tensor = self.preprocess(image)
        np.copyto(self.inputs[0]["host"], tensor.ravel())

        # H2D
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v3(self.stream.handle)
        # D2H
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()

        return self.outputs[0]["host"].reshape(self.outputs[0]["shape"])

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference and return a segmentation mask (H×W uint8).
        Output is upsampled back to the original image size.
        """
        import cv2
        h, w   = image.shape[:2]
        logits = self.infer(image)                                     # 1×C×h'×w'
        mask   = logits[0].argmax(axis=0).astype(np.uint8)             # h'×w'
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
