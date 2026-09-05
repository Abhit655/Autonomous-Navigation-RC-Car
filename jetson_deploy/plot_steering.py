"""
plot_steering.py
----------------
Produces the "before/after slew-rate limiting" figure from two telemetry CSVs.

The comparison is only meaningful if both runs saw the SAME input, so generate the logs by
replaying one recorded video twice with different settings:

    python3 autonomous_drive.py --video track1.mp4 --max-steer-step 0 --log-csv no_slew.csv
    python3 autonomous_drive.py --video track1.mp4 --max-steer-step 8 --log-csv slew.csv

Then:

    python3 plot_steering.py no_slew.csv slew.csv -o steering_comparison.png

Requires matplotlib:  pip3 install matplotlib
"""

import argparse
import csv


def load(path):
    t, cmd, actual, offset = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["t"]))
            cmd.append(float(row["steer_cmd"]))
            # Servo degrees -> normalised -1..+1 so both traces share an axis.
            # 75 = straight, 60 = full right (+1), 110 = full left (-1).
            deg = float(row["servo_actual_deg"])
            actual.append(-(deg - 75.0) / (35.0 if deg > 75 else 15.0))
            offset.append(float(row["offset"]))
    return t, cmd, actual, offset


def main():
    p = argparse.ArgumentParser(description="Plot steering traces from telemetry CSVs.")
    p.add_argument("without_limit", help="CSV logged with --max-steer-step 0")
    p.add_argument("with_limit", help="CSV logged with --max-steer-step 8")
    p.add_argument("-o", "--out", default="steering_comparison.png")
    p.add_argument("--seconds", type=float, default=None,
                   help="Plot only the first N seconds (a 10-20 s window reads best on a poster).")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0, cmd0, act0, off0 = load(args.without_limit)
    t1, cmd1, act1, off1 = load(args.with_limit)

    def clip(t, *series):
        if args.seconds is None:
            return (t,) + series
        n = sum(1 for x in t if x <= args.seconds)
        return (t[:n],) + tuple(s[:n] for s in series)

    t0, cmd0, act0, off0 = clip(t0, cmd0, act0, off0)
    t1, cmd1, act1, off1 = clip(t1, cmd1, act1, off1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t0, act0, lw=1.0, color="#c0392b")
    axes[0].set_title("Without slew-rate limiting", loc="left", fontsize=11)
    axes[0].set_ylabel("steering  (-1 left  ->  +1 right)")

    axes[1].plot(t1, act1, lw=1.0, color="#1e8449")
    axes[1].set_title("With slew-rate limiting (8°/cycle)", loc="left", fontsize=11)
    axes[1].set_ylabel("steering  (-1 left  ->  +1 right)")
    axes[1].set_xlabel("time (s)")

    for ax in axes:
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_ylim(-1.15, 1.15)
        ax.grid(alpha=0.25)

    # Quantify it: mean absolute change per control cycle is a clean single number for a caption.
    def jerk(series):
        if len(series) < 2:
            return 0.0
        return sum(abs(series[i] - series[i - 1]) for i in range(1, len(series))) / (len(series) - 1)

    j0, j1 = jerk(act0), jerk(act1)
    fig.suptitle(
        f"Steering command over identical recorded input\n"
        f"mean change per cycle: {j0:.3f} without limiting  ->  {j1:.3f} with  "
        f"({(1 - j1 / j0) * 100:.0f}% reduction)" if j0 > 0 else "Steering command",
        fontsize=12)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")
    print(f"Mean |change| per cycle:  without={j0:.4f}  with={j1:.4f}")


if __name__ == "__main__":
    main()
