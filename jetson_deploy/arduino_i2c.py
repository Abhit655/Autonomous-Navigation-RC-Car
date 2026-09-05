"""
arduino_i2c.py
--------------
Actuation driver: talks to the Arduino Uno (I2C slave at 0x40 on /dev/i2c-7) that generates
the PWM for the steering servo (D3) and the motor ESC (D6).

Wire protocol - 6 byte payload written to register 0, each field big-endian high/low:

    [devH, devL, valH, valL, msH, msL]

    device   0 = steering, 1 = motor
    value    steering 60..110 (Servo.write degrees), motor 1000..2000 (ESC microseconds)
    duration hold time in milliseconds before the Arduino auto-returns to neutral

That duration field is the safety watchdog: if this script dies or hangs, the Arduino
returns the car to neutral on its own within the hold time. Control loops must therefore
re-send faster than the duration expires, or the servo will visibly snap back to centre
between commands (Sean's thesis hit exactly this - he settled on 125 ms).

MEASURED GEOMETRY ON THIS CAR:
    steering  75 = straight, 110 = full LEFT, 60 = full RIGHT
    Note the travel is asymmetric: +35 degrees of left, only -15 of right. Steering commands
    are therefore mapped per-direction rather than with one linear scale, otherwise a
    full-lock left command would be more than twice as strong as full-lock right.

    motor     1500 = neutral, 1600 = confirmed working forward, 1400 = reverse

POWER-ON ORDER (the ESC arms once, in the Arduino's setup()):
    1. drive pack on
    2. reset the Arduino
    3. wait 5 seconds
  Connecting the pack after the Arduino has booted means the ESC never sees the arming
  pulse and will not respond.
"""

import time

try:
    import smbus2 as smbus
except ImportError:
    import smbus


# --- Protocol constants -------------------------------------------------------------
I2C_BUS = 7
I2C_ADDR = 0x40
REGISTER = 0

DEV_STEERING = 0
DEV_MOTOR = 1

# Steering, in Servo.write() degrees (NOT microseconds).
STEER_NEUTRAL = 75
STEER_FULL_LEFT = 110
STEER_FULL_RIGHT = 60

# Motor, in ESC microseconds.
MOTOR_NEUTRAL = 1500
MOTOR_MIN = 1000
MOTOR_MAX = 2000


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class CarActuator:
    """Steering + throttle over I2C, with clamping and a guaranteed neutral on exit."""

    def __init__(self, bus_num=I2C_BUS, addr=I2C_ADDR, hold_ms=150,
                 steer_neutral=STEER_NEUTRAL,
                 steer_full_left=STEER_FULL_LEFT,
                 steer_full_right=STEER_FULL_RIGHT,
                 motor_max=1600,
                 motor_creep=1550,
                 max_steer_deg_per_step=8.0):
        """
        hold_ms      Watchdog duration sent with every command. The control loop must run
                     faster than this (10 Hz loop with 150 ms hold gives comfortable overlap).
        motor_max    Upper limit this driver will ever command. Deliberately defaults to 1600
                     (Sean's confirmed working forward value) rather than the ESC's 2000, so a
                     bug cannot send the car off at full throttle.

        motor_creep  DEADBAND COMPENSATION. Brushless ESCs ignore commands close to neutral,
                     so a naive 1500..motor_max mapping wastes most of the throttle range in a
                     dead zone: a PID asking for 0.20 throttle would send 1520 us and the car
                     would simply not move. Instead, any throttle > 0 is mapped onto
                     motor_creep..motor_max, so the smallest positive command already produces
                     motion. Exactly 0.0 still sends true neutral (1500) and stops the car.

                     Default 1550 is a guess - find the real value by stepping the ESC up from
                     1520 on a stand and noting where the wheels first turn reliably.

        max_steer_deg_per_step
                     SLEW RATE LIMIT: the most the servo is allowed to move, in degrees, per
                     call to set_steering(). Without this a single noisy frame can command a
                     full lock-to-lock swing in one control cycle - which is exactly what the
                     bench logs showed (steer flipping +1.000 / -1.000 between consecutive
                     frames as the heading estimate changed sign).

                     At 21 Hz, 8 deg/step still allows the full 50 deg of travel in about
                     0.3 s, so the car stays responsive while becoming physically incapable
                     of snapping lock-to-lock on one bad detection.

                     Set to 0 or None to disable the limit.
        """
        self.bus = smbus.SMBus(bus_num)
        self.addr = addr
        self.hold_ms = int(hold_ms)

        self.steer_neutral = steer_neutral
        self.steer_full_left = steer_full_left
        self.steer_full_right = steer_full_right
        self.motor_max = motor_max
        self.motor_creep = motor_creep
        self.max_steer_deg_per_step = max_steer_deg_per_step

        self._last_steer_value = steer_neutral
        self._last_motor_value = MOTOR_NEUTRAL
        self._commanded_steer_deg = float(steer_neutral)  # pre-slew target, for telemetry

        # I2C robustness counters (see _send).
        self.i2c_retries = 3
        self.i2c_failure_warn_after = 10
        self.i2c_failures = 0
        self.consecutive_i2c_failures = 0

    # --- low level ------------------------------------------------------------------
    def _send(self, device, value, duration_ms):
        """
        Write one command, retrying on transient I2C failures.

        OSError 121 ("remote I/O error") means the Arduino did not ACK. On a moving RC car
        this happens occasionally from electrical noise (the motor is right next to the bus),
        a marginal SDA/SCL/ground connection, or the Arduino being busy. A single dropped
        frame is harmless - the Arduino simply holds its previous command until the watchdog
        expires - so retry briefly and carry on rather than crashing the control loop and
        leaving the car driving.
        """
        payload = [
            (device >> 8) & 0xFF, device & 0xFF,
            (value >> 8) & 0xFF, value & 0xFF,
            (duration_ms >> 8) & 0xFF, duration_ms & 0xFF,
        ]

        for attempt in range(self.i2c_retries):
            try:
                self.bus.write_i2c_block_data(self.addr, REGISTER, payload)
                self.consecutive_i2c_failures = 0
                return True
            except OSError:
                if attempt < self.i2c_retries - 1:
                    time.sleep(0.002)

        self.i2c_failures += 1
        self.consecutive_i2c_failures += 1

        # A sustained outage is different from an occasional glitch: the Arduino has probably
        # lost power or a wire has come off. Surface it loudly, but still do not crash - the
        # Arduino's own watchdog will have neutralled the car already.
        if self.consecutive_i2c_failures == self.i2c_failure_warn_after:
            print(f"WARNING: {self.consecutive_i2c_failures} consecutive I2C failures - "
                  f"check Arduino power and the SDA/SCL/GND wiring.")
        return False

    # --- raw units ------------------------------------------------------------------
    def set_steering_raw(self, degrees, duration_ms=None):
        """Command the servo directly in its own degree units (60..110)."""
        v = int(_clamp(degrees, STEER_FULL_RIGHT, STEER_FULL_LEFT))
        self._send(DEV_STEERING, v, duration_ms or self.hold_ms)
        self._last_steer_value = v
        return v

    def set_motor_raw(self, microseconds, duration_ms=None):
        """Command the ESC directly in microseconds (clamped to motor_max for safety)."""
        v = int(_clamp(microseconds, MOTOR_MIN, self.motor_max))
        self._send(DEV_MOTOR, v, duration_ms or self.hold_ms)
        self._last_motor_value = v
        return v

    # --- normalised units (what the controller speaks) ------------------------------
    def set_steering(self, steer, duration_ms=None):
        """
        steer: -1.0 (full left) .. 0.0 (straight) .. +1.0 (full right)

        Mapped per-direction because the mechanical travel is asymmetric on this car:
        35 degrees available to the left of neutral, only 15 to the right.
        """
        steer = _clamp(steer, -1.0, 1.0)
        if steer >= 0.0:
            span = self.steer_neutral - self.steer_full_right   # 75 -> 60 = 15
            degrees = self.steer_neutral - steer * span
        else:
            span = self.steer_full_left - self.steer_neutral    # 75 -> 110 = 35
            degrees = self.steer_neutral + (-steer) * span

        self._commanded_steer_deg = degrees

        # Slew limit: move at most max_steer_deg_per_step toward the target this cycle.
        if self.max_steer_deg_per_step:
            delta = degrees - self._last_steer_value
            limit = self.max_steer_deg_per_step
            if delta > limit:
                degrees = self._last_steer_value + limit
            elif delta < -limit:
                degrees = self._last_steer_value - limit

        return self.set_steering_raw(degrees, duration_ms)

    @property
    def steer_target_vs_actual(self):
        """(pre-slew target, post-slew actual) in servo degrees - useful for telemetry."""
        return self._commanded_steer_deg, self._last_steer_value

    def set_throttle(self, throttle, duration_ms=None):
        """
        throttle: 0.0 (stopped) .. 1.0 (motor_max). Negative values are ignored - this
        vehicle is a lane follower and never needs reverse under autonomous control.

        Mapped over motor_creep..motor_max rather than neutral..motor_max so that small
        throttle values actually move the car instead of landing in the ESC's dead zone.
        """
        throttle = _clamp(throttle, 0.0, 1.0)
        if throttle <= 0.0:
            us = MOTOR_NEUTRAL
        else:
            us = self.motor_creep + throttle * (self.motor_max - self.motor_creep)
        return self.set_motor_raw(us, duration_ms)

    # --- lifecycle ------------------------------------------------------------------
    def neutral(self):
        """Centre the steering and stop the motor."""
        self.set_steering_raw(self.steer_neutral)
        self.set_motor_raw(MOTOR_NEUTRAL)

    def close(self):
        """Always leave the car in a safe state, even if the caller crashed."""
        try:
            # Sent several times: this is the one message that absolutely must land, since a
            # dropped frame here would leave the car driving away.
            for _ in range(3):
                self.neutral()
                time.sleep(0.03)
            if self.i2c_failures:
                print(f"({self.i2c_failures} I2C write failures during this run)")
        finally:
            try:
                self.bus.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


# --- Manual test: python3 arduino_i2c.py  (wheels off the ground, pack disconnected) ---
if __name__ == "__main__":
    print("Steering sweep test - car should be on a stand, drive pack DISCONNECTED.")
    with CarActuator() as car:
        for label, s in [("straight", 0.0), ("half left", -0.5), ("full left", -1.0),
                         ("straight", 0.0), ("half right", 0.5), ("full right", 1.0),
                         ("straight", 0.0)]:
            deg = car.set_steering(s)
            print(f"  {label:<12} steer={s:+.2f}  -> servo {deg}")
            time.sleep(0.8)
    print("Done - neutral.")
