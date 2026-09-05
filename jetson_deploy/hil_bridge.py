"""
hil_bridge.py
-------------
Hardware-in-the-loop bridge. Unity computes steering/throttle for the simulated car and
sends them over UDP; this script receives them on the Jetson and drives the REAL servo and
ESC through the Arduino. The physical car mirrors whatever the simulated car is doing.

    Unity (Mac)  --UDP-->  hil_bridge.py (Jetson)  --I2C-->  Arduino  --PWM-->  servo + ESC

Useful for:
  * proving the whole command chain end to end without needing the track
  * seeing how the real steering geometry responds to commands tuned in simulation
  * a genuinely good demo: the simulated car laps the virtual track while the physical car
    on the bench steers in perfect sync

Packet format - 9 bytes, network byte order:
    magic  1 byte   0xA5
    steer  4 bytes  float32,  -1.0 (full left) .. +1.0 (full right)
    thr    4 bytes  float32,   0.0 .. 1.0

SAFETY
  * Car on a stand, wheels off the ground. This is a bench tool, not a driving mode.
  * --steering-only (default) never touches the motor. Drop it deliberately when you want
    the wheels to spin too.
  * If no packet arrives for --timeout seconds the car is neutralled automatically, so
    closing Unity or losing Wi-Fi stops the car rather than leaving it running.

Usage:
    python3 hil_bridge.py                      # steering only, safest
    python3 hil_bridge.py --with-throttle      # motor live as well
"""

import argparse
import socket
import struct
import sys
import time

from arduino_i2c import CarActuator

MAGIC = 0xA5
PACKET = struct.Struct("!Bff")   # magic (uint8), steer (float32), throttle (float32) = 9 bytes


def main():
    p = argparse.ArgumentParser(description="Unity -> real car HIL bridge.")
    p.add_argument("--port", type=int, default=9099)
    p.add_argument("--with-throttle", action="store_true",
                   help="Also drive the motor. Omit this and only the steering mirrors Unity.")
    p.add_argument("--timeout", type=float, default=0.5,
                   help="Seconds without a packet before the car is neutralled.")
    p.add_argument("--motor-max", type=int, default=1600)
    p.add_argument("--motor-creep", type=int, default=1550,
                   help="ESC value where the wheels first move. Throttle is mapped from here to "
                        "--motor-max so small commands are not swallowed by the deadband.")
    p.add_argument("--max-steer-step", type=float, default=8.0,
                   help="Servo slew limit in degrees per update. 0 disables.")
    p.add_argument("--hold-ms", type=int, default=150)
    args = p.parse_args()

    if not args.with_throttle:
        print("*** STEERING ONLY - the motor will not be driven. ***")
    else:
        print("*** THROTTLE LIVE - car must be on a stand, wheels clear. ***")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(0.1)

    print(f"Listening for Unity on UDP {args.port}. Ctrl+C to stop.")

    car = CarActuator(hold_ms=args.hold_ms, motor_max=args.motor_max,
                      motor_creep=args.motor_creep,
                      max_steer_deg_per_step=args.max_steer_step)

    last_packet = 0.0
    packets = 0
    last_report = time.time()
    neutralled = True

    try:
        while True:
            try:
                data, _addr = sock.recvfrom(64)
            except socket.timeout:
                data = None

            now = time.time()

            if data and len(data) == PACKET.size:
                magic, steer, throttle = PACKET.unpack(data)
                if magic == MAGIC:
                    last_packet = now
                    packets += 1
                    neutralled = False

                    deg = car.set_steering(steer)
                    if args.with_throttle:
                        car.set_throttle(throttle)
                    else:
                        car.set_throttle(0.0)

                    if now - last_report >= 1.0:
                        rate = packets / (now - last_report)
                        print(f"[{rate:5.1f} pkt/s] steer={steer:+.3f} -> servo {deg}  "
                              f"thr={throttle:.2f}"
                              f"{'' if args.with_throttle else ' (ignored)'}")
                        packets = 0
                        last_report = now

            # Link-loss watchdog: Unity stopped, or the network dropped.
            if not neutralled and (now - last_packet) > args.timeout:
                car.neutral()
                neutralled = True
                print("No packets - car neutralled. Waiting for Unity...")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        car.close()
        sock.close()
        print("Car neutral, socket closed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
