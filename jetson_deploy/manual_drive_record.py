"""
manual_drive_record.py
----------------------
Drive the car yourself from the SSH terminal, while recording camera footage, the
segmentation overlay and a telemetry CSV.

Why this exists: the Jetson is headless at the track, so anything using cv2.imshow or
keyboard libraries that need a window will not work. This reads keys directly from the
terminal in raw mode, so it works over a plain SSH session with nothing installed.

    W / S   forward / reverse
    A / D   steer left / right
    SPACE   stop immediately
    Q       quit (car neutralled)

Keys are momentary: a terminal cannot tell you when a key is RELEASED, only when it is
pressed. So each press applies for a short window and then decays back to neutral. Hold the
key down and the OS auto-repeat keeps it alive - exactly like holding a throttle trigger.

Use this to capture real track footage. Recorded video can be replayed offline through the
same controller to tune gains without the car:

    python3 autonomous_drive.py --video lap1.mp4 --kp 0.7 --log-csv test.csv

SAFETY
  * Same power sequence as always: pack on, reset Arduino, wait 5 s.
  * --no-throttle records and steers but never drives, for a first pass.
  * Release everything or press SPACE to stop. Q exits cleanly and neutrals the car.
"""

import argparse
import csv
import os
import select
import sys
import termios
import time
import tty

import cv2
import numpy as np

from arduino_i2c import CarActuator
from autonomous_drive import SegFormerTRT, gstreamer_pipeline, colorize
from mjpeg_server import MjpegStreamer


def main():
    p = argparse.ArgumentParser(description="Manual keyboard driving with recording.")
    p.add_argument("--engine",
                   default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "models", "exported", "segformer_b0_orin_fp16.engine"))
    p.add_argument("--flip-method", type=int, default=2)
    p.add_argument("--capture-width", type=int, default=1280)
    p.add_argument("--capture-height", type=int, default=720)
    p.add_argument("--frame-width", type=int, default=640)
    p.add_argument("--frame-height", type=int, default=480)

    p.add_argument("--record", default="manual_lap.mp4", help="Output video path.")
    p.add_argument("--log-csv", default=None, help="Optional telemetry CSV.")
    p.add_argument("--stream-port", type=int, default=0, help="Live browser view, e.g. 8080.")
    p.add_argument("--raw", action="store_true",
                   help="Record the plain camera feed with no segmentation overlay. Use this if the "
                        "footage is intended for annotating and retraining.")
    p.add_argument("--no-segmentation", action="store_true",
                   help="Skip inference entirely - records faster, but no overlay or telemetry.")

    p.add_argument("--no-throttle", action="store_true", help="Steering only, motor never driven.")
    p.add_argument("--motor-creep", type=int, default=1590)
    p.add_argument("--motor-max", type=int, default=1600)
    p.add_argument("--steer-step", type=float, default=0.35,
                   help="How much one key press moves the steering, 0-1.")
    p.add_argument("--throttle-step", type=float, default=0.5,
                   help="Throttle applied while W is held, 0-1.")
    p.add_argument("--key-hold", type=float, default=0.25,
                   help="Seconds a key press stays active before decaying back to neutral.")
    p.add_argument("--rate", type=float, default=20.0)
    args = p.parse_args()

    cap = cv2.VideoCapture(gstreamer_pipeline(
        args.capture_width, args.capture_height, 30, args.flip_method,
        args.frame_width, args.frame_height), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("ERROR: camera would not open. If a previous script crashed, run:\n"
              "  sudo systemctl restart nvargus-daemon", file=sys.stderr)
        return 1

    ok, frame = cap.read()
    if not ok:
        print("ERROR: camera opened but gave no frame.", file=sys.stderr)
        return 1
    print(f"Camera OK: {frame.shape[1]}x{frame.shape[0]}")

    seg = None
    if not args.no_segmentation:
        seg = SegFormerTRT(args.engine)

    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.rate, (args.frame_width, args.frame_height))
    print(f"Recording to {args.record}")

    csv_file = csv_writer = None
    if args.log_csv:
        csv_file = open(args.log_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["t", "frame", "steer", "throttle", "servo_deg"])

    streamer = MjpegStreamer(port=args.stream_port) if args.stream_port else None
    if streamer:
        print(f"Live view: http://<jetson-ip>:{args.stream_port}/")

    car = CarActuator(motor_creep=args.motor_creep, motor_max=args.motor_max,
                      max_steer_deg_per_step=99)   # no slew limit - you are the controller

    if args.no_throttle:
        print("*** STEERING ONLY - motor will not be driven. ***")

    print("\n  W/S drive   A/D steer   SPACE stop   Q quit\n")

    old_term = termios.tcgetattr(sys.stdin)
    period = 1.0 / max(1.0, args.rate)
    steer = throttle = 0.0
    last_steer_key = last_throttle_key = 0.0
    frames = 0
    t_start = time.time()

    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            loop_start = time.time()

            # --- read every key waiting in the buffer (auto-repeat sends many) ---
            while select.select([sys.stdin], [], [], 0)[0]:
                c = sys.stdin.read(1).lower()
                now = time.time()
                if c == "q":
                    raise KeyboardInterrupt
                elif c == "a":
                    steer = -args.steer_step; last_steer_key = now
                elif c == "d":
                    steer = args.steer_step; last_steer_key = now
                elif c == "w":
                    throttle = args.throttle_step; last_throttle_key = now
                elif c == "s":
                    throttle = -args.throttle_step; last_throttle_key = now
                elif c == " ":
                    steer = throttle = 0.0

            # --- decay back to neutral when keys stop repeating ---
            now = time.time()
            if now - last_steer_key > args.key_hold:
                steer = 0.0
            if now - last_throttle_key > args.key_hold:
                throttle = 0.0

            ok, frame = cap.read()
            if not ok:
                continue

            servo_deg = car.set_steering(steer)
            car.set_throttle(0.0 if (args.no_throttle or throttle <= 0) else throttle)

            # --- build the frame to save ---
            out = frame
            telem = f"steer={steer:+.2f} thr={throttle:+.2f} servo={servo_deg}"
            if seg is not None and not args.raw:
                mask = seg.infer(frame)
                out = cv2.addWeighted(
                    cv2.resize(colorize(mask), (frame.shape[1], frame.shape[0]),
                               interpolation=cv2.INTER_NEAREST), 0.55, frame, 0.45, 0)
            elif seg is not None:
                out = frame.copy()

            if not args.raw:
                cv2.putText(out, telem, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(out)
            if streamer:
                streamer.publish(out)

            frames += 1
            if csv_writer:
                csv_writer.writerow([f"{time.time()-t_start:.4f}", frames,
                                     f"{steer:.4f}", f"{throttle:.4f}", f"{servo_deg}"])

            if frames % 20 == 0:
                fps = frames / (time.time() - t_start)
                sys.stdout.write(f"\r  [{fps:5.1f} fps] {telem}   ")
                sys.stdout.flush()

            sleep = period - (time.time() - loop_start)
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        car.close()
        cap.release()
        writer.release()
        if csv_file:
            csv_file.close()
        if streamer:
            streamer.close()
        print(f"Car neutral. Saved {args.record}"
              + (f" and {args.log_csv}" if args.log_csv else ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
