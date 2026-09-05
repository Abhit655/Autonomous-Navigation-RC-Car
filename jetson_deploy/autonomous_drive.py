"""
autonomous_drive.py
-------------------
Main autonomous loop on the Jetson Orin Nano:

    CSI camera -> SegFormer (TensorRT) -> class mask -> PID/Kalman controller -> Arduino (I2C)

Runs headless - there is no display attached over SSH, so status is printed to the console
and an annotated video can optionally be recorded for review afterwards.

SAFETY - read before the first powered run:
  * Always start with --dry-run. Steering still moves so you can watch it react, but no
    throttle is ever sent.
  * Then run with the car ON A STAND, wheels off the ground, before ever putting it down.
  * The Arduino's duration watchdog returns the car to neutral ~150 ms after the last
    command, so a crashed script stops the car. Pulling the drive pack is the hard kill.
  * Power order: drive pack on -> reset Arduino -> wait 5 s (the ESC arms once, in setup()).

Examples:
    # perception + control maths only, no motor
    python3 autonomous_drive.py --dry-run

    # on a stand, slow
    python3 autonomous_drive.py --base-throttle 0.3

    # record what it saw for offline review
    python3 autonomous_drive.py --dry-run --record run1.mp4
"""

import argparse
import csv
import os
import signal
import sys
import time

import cv2
import numpy as np
import tensorrt as trt

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401  (initialises the CUDA context on import)
except ImportError:
    print("ERROR: pycuda is not installed.\n"
          "  sudo apt-get install -y python3-libnvinfer python3-pycuda\n"
          "  (or) pip3 install --no-cache-dir pycuda", file=sys.stderr)
    raise

from arduino_i2c import CarActuator
from lane_pid_controller import LanePidController
from mjpeg_server import MjpegStreamer


# Preprocessing must match training exactly (see scripts/train.py and infer_unity_frames.py).
INPUT_SIZE = 640
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = ["background", "road", "divider-line", "dotted-line",
               "double-line", "random-line", "road-sign-line", "solid-line"]

CLASS_COLORS_BGR = {
    0: (0, 0, 0), 1: (200, 200, 200), 2: (200, 0, 200), 3: (0, 0, 255),
    4: (0, 140, 255), 5: (0, 255, 255), 6: (255, 0, 0), 7: (0, 255, 0),
}


# ---------------------------------------------------------------------------------------
def gstreamer_pipeline(capture_width=1280, capture_height=720, framerate=30,
                       flip_method=2, out_width=640, out_height=480):
    """CSI camera (Raspberry Pi Camera V2 / IMX219) via nvarguscamerasrc."""
    return (
        f"nvarguscamerasrc sensor-id=0 ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={out_width}, height={out_height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


class SegFormerTRT:
    """TensorRT 10 inference wrapper. Note the v10 API (get_tensor_name / execute_async_v3)."""

    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = tuple(self.engine.get_tensor_shape(name))
            else:
                self.output_name = name
                self.output_shape = tuple(self.engine.get_tensor_shape(name))

        self.h_input = cuda.pagelocked_empty(int(np.prod(self.input_shape)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(int(np.prod(self.output_shape)), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        print(f"Engine loaded: {self.input_name}{self.input_shape} -> "
              f"{self.output_name}{self.output_shape}")

    def infer(self, frame_bgr):
        """frame_bgr -> (H, W) uint8 class-ID mask at the model's native output resolution."""
        resized = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = ((rgb - MEAN) / STD).transpose(2, 0, 1)

        np.copyto(self.h_input, chw.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        logits = self.h_output.reshape(self.output_shape)      # (1, 8, 160, 160)
        return np.argmax(logits[0], axis=0).astype(np.uint8)   # (160, 160)


class _NullActuator:
    """
    Stand-in for CarActuator during --video replay: applies the same slew limit so the
    steering trace you see matches what the real car would have done, but sends nothing.
    """

    def __init__(self, max_steer_deg_per_step=8.0):
        self.max_steer_deg_per_step = max_steer_deg_per_step
        self._last = 75.0
        self._target = 75.0

    def set_steering(self, steer):
        steer = max(-1.0, min(1.0, steer))
        deg = 75 - steer * 15 if steer >= 0 else 75 + (-steer) * 35
        self._target = deg
        if self.max_steer_deg_per_step:
            d = deg - self._last
            lim = self.max_steer_deg_per_step
            deg = self._last + (lim if d > lim else (-lim if d < -lim else d))
        self._last = deg
        return deg

    def set_throttle(self, throttle):
        return 1500

    @property
    def steer_target_vs_actual(self):
        return self._target, self._last

    def close(self):
        pass


def colorize(mask):
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cid, colour in CLASS_COLORS_BGR.items():
        if cid == 0:
            continue
        out[mask == cid] = colour
    return out


# ---------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Autonomous lane following on the Jetson.")
    p.add_argument("--engine",
                   default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "models", "exported", "segformer_b0_orin_fp16.engine"),
                   help="TensorRT engine. Build it for your device first - see training/build_engine.py.")
    p.add_argument("--flip-method", type=int, default=2)
    p.add_argument("--capture-width", type=int, default=1280)
    p.add_argument("--capture-height", type=int, default=720)
    p.add_argument("--frame-width", type=int, default=640, help="Frame size fed to the model/controller.")
    p.add_argument("--frame-height", type=int, default=480)
    p.add_argument("--usb", action="store_true", help="Use /dev/video0 as a plain V4L2 camera instead of CSI.")
    p.add_argument("--video", default=None,
                   help="Replay an .mp4 instead of the live camera. Implies no actuation - use this "
                        "to tune gains offline against footage recorded at the track, which is far "
                        "faster and safer than tuning on a moving car.")
    p.add_argument("--loop", action="store_true", help="With --video, restart when the file ends.")
    p.add_argument("--speed", type=float, default=1.0,
                   help="With --video, playback speed multiplier. Use a high value to sweep a long "
                        "clip quickly, or 0 to run as fast as possible.")

    p.add_argument("--dry-run", action="store_true",
                   help="Never send throttle. Steering still moves so you can watch it react.")
    p.add_argument("--rate", type=float, default=21.0,
                   help="Control loop rate in Hz. 21 is the measured SegFormer ceiling on the Orin Nano; "
                        "asking for more just idles.")
    p.add_argument("--hold-ms", type=int, default=150,
                   help="Arduino auto-neutral watchdog. Must exceed the loop period.")
    p.add_argument("--motor-max", type=int, default=1600, help="Hard cap on ESC microseconds.")
    p.add_argument("--motor-creep", type=int, default=1590,
                   help="ESC value where the wheels first move. Throttle maps from here to "
                        "--motor-max, so small commands clear the deadband instead of doing nothing.")

    # Controller gains - same names and defaults as the Unity component.
    # Defaults below are the values validated in the Unity digital twin, where this controller
    # held the lane through curves and across crossing/parallel lines. Start here on the track
    # and adjust from what the car actually does, not from scratch.
    p.add_argument("--kp", type=float, default=0.856)
    p.add_argument("--ki", type=float, default=0.36)
    p.add_argument("--kd", type=float, default=0.55)
    p.add_argument("--heading-gain", type=float, default=0.0,
                   help="Zero by design: far-field scanning makes the offset measurement inherently "
                        "anticipatory, which made the explicit feed-forward term redundant.")
    p.add_argument("--offset-trim", type=float, default=0.0,
                   help="Corrects a systematic left/right bias. Set this on the track once the "
                        "oscillation is settled - not before.")
    p.add_argument("--base-throttle", type=float, default=0.15)
    p.add_argument("--corner-slowdown", type=float, default=0.7)
    p.add_argument("--lost-line-throttle", type=float, default=0.0,
                   help="Throttle while no lines are visible. 0 (stop) on the real car - the sim used "
                        "a crawl, but stopping is the right default with real hardware on a real track.")
    p.add_argument("--crop-top", type=float, default=0.0,
                   help="Fraction of the mask top blanked before the controller sees it. Only useful "
                        "for the RL agent, which has no scan band of its own - leave at 0 for the PID, "
                        "whose scan band already decides where to look.")
    p.add_argument("--scan-band-top", type=float, default=0.0,
                   help="Top of the scanned region, as a fraction of image height (0 = very top). "
                        "Far-field scanning gives curve anticipation.")
    p.add_argument("--scan-band-bottom", type=float, default=0.5,
                   help="Bottom of the scanned region. 0.0-0.5 scans the far field, which was found "
                        "to hold the lane far better through curves than near-field scanning.")
    p.add_argument("--min-lane-width", type=int, default=18,
                   help="Reject rows whose detected lane is narrower than this many mask pixels - "
                        "these are almost always the two edges of a single line.")
    p.add_argument("--steering-deadband", type=float, default=0.02,
                   help="Ignore offsets smaller than this, so mask noise does not cause micro-steering.")

    p.add_argument("--record", default=None, help="Optional .mp4 path to save an annotated video.")
    p.add_argument("--log-csv", default=None,
                   help="Write per-cycle telemetry to a CSV. Combine with --video replay to compare "
                        "two configurations against IDENTICAL input - e.g. slew limiting on vs off.")
    p.add_argument("--stream-port", type=int, default=0,
                   help="If set (e.g. 8080), serve a live MJPEG view at http://<jetson-ip>:PORT/ "
                        "Watch from any browser - no software needed on the viewing device.")
    p.add_argument("--stream-scale", type=float, default=0.75,
                   help="Downscale factor for the streamed image. Lower = less bandwidth.")
    p.add_argument("--max-steer-step", type=float, default=8.0,
                   help="Slew limit: max servo degrees of movement per control cycle. Prevents a "
                        "single noisy frame commanding a lock-to-lock swing. 0 disables.")
    args = p.parse_args()

    # Replaying a file is inherently offline: never actuate from recorded footage.
    replay = args.video is not None
    if replay:
        args.dry_run = True
        print(f"*** REPLAY MODE: {args.video} - no I2C, no motor, no steering. ***")
    elif args.dry_run:
        print("*** DRY RUN - no throttle will be sent. Steering is live. ***")

    # --- camera / video source ---
    if replay:
        cap = cv2.VideoCapture(args.video)
    elif args.usb:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)
    else:
        cap = cv2.VideoCapture(gstreamer_pipeline(
            args.capture_width, args.capture_height, 30, args.flip_method,
            args.frame_width, args.frame_height), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("ERROR: could not open the camera. Try --usb, or check the GStreamer pipeline.",
              file=sys.stderr)
        return 1

    ok, frame = cap.read()
    if not ok:
        print("ERROR: camera opened but returned no frame.", file=sys.stderr)
        return 1
    print(f"Camera OK: {frame.shape[1]}x{frame.shape[0]}")

    seg = SegFormerTRT(args.engine)

    controller = LanePidController(
        scan_band_top=args.scan_band_top,
        scan_band_bottom=args.scan_band_bottom,
        min_lane_width_px=args.min_lane_width,
        kp=args.kp, ki=args.ki, kd=args.kd,
        heading_gain=args.heading_gain,
        offset_trim=args.offset_trim,
        steering_deadband=args.steering_deadband,
        base_throttle=args.base_throttle,
        corner_slowdown=args.corner_slowdown,
        lost_line_throttle=args.lost_line_throttle,
    )
    print(f"Controller: scan {args.scan_band_top}-{args.scan_band_bottom}  "
          f"kp={args.kp} ki={args.ki} kd={args.kd} heading={args.heading_gain}  "
          f"throttle={args.base_throttle} creep={args.motor_creep} max={args.motor_max}")

    writer = None
    if args.record:
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.rate, (args.frame_width, args.frame_height))

    running = {"go": True}

    def stop(signum, _frame):
        print("\nStopping...")
        running["go"] = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    csv_file = None
    csv_writer = None
    if args.log_csv:
        csv_file = open(args.log_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "t", "frame", "offset", "heading", "steer_cmd",
            "servo_target_deg", "servo_actual_deg", "throttle",
            "rows_found", "lane_width_px", "lines_visible",
        ])
        print(f"Logging telemetry to {args.log_csv}")

    streamer = None
    if args.stream_port:
        streamer = MjpegStreamer(port=args.stream_port, scale=args.stream_scale)
        print(f"Live view: http://<jetson-ip>:{args.stream_port}/")

    period = 1.0 / max(1e-3, args.rate)
    if replay and args.speed != 1.0:
        period = 0.0 if args.speed <= 0 else period / args.speed

    # In replay mode there is no car - a null actuator keeps the loop below identical.
    if replay:
        car = _NullActuator()
    else:
        car = CarActuator(hold_ms=args.hold_ms, motor_max=args.motor_max,
                          motor_creep=args.motor_creep,
                          max_steer_deg_per_step=args.max_steer_step)

    print("Running. Ctrl+C to stop.")
    last = time.time()
    frames = 0
    t_start = time.time()

    try:
        while running["go"]:
            loop_start = time.time()

            ok, frame = cap.read()
            if not ok:
                if replay:
                    if args.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print("End of video.")
                    break
                print("frame grab failed", file=sys.stderr)
                continue

            if replay and (frame.shape[1] != args.frame_width or frame.shape[0] != args.frame_height):
                frame = cv2.resize(frame, (args.frame_width, args.frame_height))

            mask = seg.infer(frame)

            # Match the Unity observation crop: ignore the top band (horizon clutter,
            # distant lanes, stand/background false positives).
            if args.crop_top > 0:
                crop_rows = int(args.crop_top * mask.shape[0])
                mask[:crop_rows, :] = 0

            now = time.time()
            dt = now - last
            last = now

            steer, throttle = controller.step(mask, dt)

            car.set_steering(steer)
            car.set_throttle(0.0 if args.dry_run else throttle)

            frames += 1

            if csv_writer is not None:
                target_deg, actual_deg = car.steer_target_vs_actual
                csv_writer.writerow([
                    f"{time.time() - t_start:.4f}", frames,
                    f"{controller.offset:.5f}", f"{controller.heading:.5f}",
                    f"{controller.steer:.5f}",
                    f"{target_deg:.2f}", f"{actual_deg:.2f}",
                    f"{controller.throttle:.4f}",
                    controller.rows_found, f"{controller.lane_width_px:.2f}",
                    int(controller.lines_visible),
                ])

            if frames % 10 == 0:
                fps = frames / (time.time() - t_start)
                print(f"[{fps:5.1f} fps] {controller.telemetry()}")

            if writer is not None or streamer is not None:
                overlay = cv2.addWeighted(
                    cv2.resize(colorize(mask), (frame.shape[1], frame.shape[0]),
                               interpolation=cv2.INTER_NEAREST), 0.55, frame, 0.45, 0)

                # Show WHERE the controller is actually looking. The mask covers the whole frame,
                # but only rows inside the scan band contribute to the offset measurement - without
                # drawing it, the overlay misleadingly suggests the whole image is being used.
                h_ov, w_ov = overlay.shape[:2]
                y_top = int(args.scan_band_top * h_ov)
                y_bot = int(args.scan_band_bottom * h_ov)

                dim = overlay.copy()
                cv2.rectangle(dim, (0, 0), (w_ov, y_top), (0, 0, 0), -1)
                cv2.rectangle(dim, (0, y_bot), (w_ov, h_ov), (0, 0, 0), -1)
                overlay = cv2.addWeighted(dim, 0.55, overlay, 0.45, 0)

                cv2.rectangle(overlay, (1, y_top), (w_ov - 2, y_bot), (0, 200, 255), 2)
                cv2.putText(overlay, "scan band", (8, max(14, y_top + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

                cv2.putText(overlay, controller.telemetry(), (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                # Steering bar: target (pre-slew) vs actual (post-slew), so you can see the
                # rate limiter working and spot saturation at a glance.
                target_deg, actual_deg = car.steer_target_vs_actual
                h, w = overlay.shape[:2]
                cx, y0 = w // 2, h - 18
                cv2.line(overlay, (cx - 100, y0), (cx + 100, y0), (90, 90, 90), 2)
                cv2.line(overlay, (cx, y0 - 6), (cx, y0 + 6), (200, 200, 200), 1)
                # servo 60..110 maps to +100..-100 px (60 = right, 110 = left)
                to_px = lambda d: int(cx - (d - 75) / 35.0 * 100)
                cv2.circle(overlay, (to_px(target_deg), y0), 4, (120, 120, 255), -1)
                cv2.circle(overlay, (to_px(actual_deg), y0), 6, (0, 255, 0), 2)
                cv2.putText(overlay, f"thr {controller.throttle:.2f}", (cx + 110, y0 + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                if writer is not None:
                    writer.write(overlay)
                if streamer is not None:
                    streamer.publish(overlay)

            sleep = period - (time.time() - loop_start)
            if sleep > 0:
                time.sleep(sleep)

    finally:
        car.close()          # always neutral, even on a crash
        cap.release()
        if writer is not None:
            writer.release()
        if streamer is not None:
            streamer.close()
        if csv_file is not None:
            csv_file.close()
            print(f"Telemetry written to {args.log_csv}")
        print("Car neutral, camera released.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
