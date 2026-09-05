"""
build_engine.py
---------------
Convert an ONNX model to a TensorRT FP16 engine.
Run this script ON the Jetson Nano (not the training machine).

Requirements (pre-installed on JetPack):
    tensorrt, pycuda

Usage:
    python deployment/tensorrt/build_engine.py --onnx models/exported/segformer_b0.onnx
    python deployment/tensorrt/build_engine.py --onnx models/exported/segformer_b0.onnx --int8
"""

import argparse
from pathlib import Path

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path: str, engine_path: str, use_fp16: bool, use_int8: bool,
                 workspace_gb: int = 1):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

        if use_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("  Precision: INT8")
        elif use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  Precision: FP16")
        else:
            print("  Precision: FP32")

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  ONNX parse error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")

        print(f"  Network inputs  : {network.num_inputs}")
        print(f"  Network outputs : {network.num_outputs}")

        # Dynamic batch dim needs an explicit optimization profile.
        # Inference is always single-frame, so fix batch=1.
        input_tensor = network.get_input(0)
        _, c, h, w = input_tensor.shape
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, (1, c, h, w), (1, c, h, w), (1, c, h, w))
        config.add_optimization_profile(profile)
        print(f"  Optimization profile: {input_tensor.name} fixed at (1, {c}, {h}, {w})")

        # Build engine
        print("\n  Building TensorRT engine (this may take several minutes)...")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("Failed to build TensorRT engine")

        Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized)

        print(f"\n✓ TensorRT engine saved → {engine_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx",         default="models/exported/segformer_b0.onnx")
    p.add_argument("--engine",       default="models/exported/segformer_b0.trt")
    p.add_argument("--fp16",         action="store_true", default=True)
    p.add_argument("--int8",         action="store_true")
    p.add_argument("--workspace-gb", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 55)
    print("  ONNX → TensorRT Engine Builder")
    print("=" * 55)
    print(f"  ONNX   : {args.onnx}")
    print(f"  Engine : {args.engine}")

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        use_fp16=args.fp16 and not args.int8,
        use_int8=args.int8,
        workspace_gb=args.workspace_gb,
    )
