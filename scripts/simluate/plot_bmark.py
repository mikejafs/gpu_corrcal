"""
Plot the output of benchmark_calibration.py.

python plot_benchmark_calibration.py                      # reads benchmark_calibration.npz
python plot_benchmark_calibration.py benchmark_calibration.npz -o fig.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

STYLE = {"cpu": dict(color="C0", marker="o", label="CPU (corrcal)"),
         "gpu": dict(color="C1", marker="s", label="GPU (gpu_corrcal)")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", type=Path, default=Path("benchmark_calibration.npz"))
    ap.add_argument("-o", "--out", type=Path, default=None, help="output image (default: <npz stem>.png)")
    args = ap.parse_args()

    f = np.load(args.npz, allow_pickle=True)
    rows = sorted(f["rows"], key=lambda r: r["n_ant"])
    backends = [str(b) for b in f["backends"]]
    cfg = f["config"].item()
    threads = cfg.get("cpu_threads", "?")

    n_ant = np.array([r["n_ant"] for r in rows])
    get = lambda b, k: np.array([r[f"{b}_{k}"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) time to the common accuracy target
    ax = axes[0]
    for b in backends:
        ax.loglog(n_ant, get(b, "t_target"), ls="-", **STYLE[b])
    ax.set_xlabel("number of antennas"); ax.set_ylabel("time to target [s]")
    ax.set_title("Calibration time to noise-limited accuracy")

    # (b) per-call cost
    ax = axes[1]
    for b in backends:
        st = dict(STYLE[b]); lab = st.pop("label")
        ax.loglog(n_ant, 1e3 * get(b, "t_nll"), ls="-", label=f"{lab}: nll", **st)
        ax.loglog(n_ant, 1e3 * get(b, "t_grad"), ls="--", mfc="none", label=f"{lab}: grad", **st)
    ax.set_xlabel("number of antennas"); ax.set_ylabel("time per call [ms]")
    ax.set_title("Objective and gradient, one call")

    # (c) speedup
    ax = axes[2]
    if {"cpu", "gpu"} <= set(backends):
        sp_target = get("cpu", "t_target") / get("gpu", "t_target")
        sp_nll = get("cpu", "t_nll") / get("gpu", "t_nll")
        sp_grad = get("cpu", "t_grad") / get("gpu", "t_grad")
        ax.loglog(n_ant, sp_target, "k-o", label="time to target")
        ax.loglog(n_ant, sp_nll, "-^", color="C2", label="nll call")
        ax.loglog(n_ant, sp_grad, "--v", color="C3", mfc="none", label="grad call")
        ax.axhline(1, color="0.6", lw=0.8)
        for x, y in zip(n_ant, sp_target):
            ax.annotate(f"{y:.1f}×" if y < 10 else f"{y:.0f}×", (x, y), textcoords="offset points",
                        xytext=(-6, -14), ha="right", fontsize=9)
        ax.set_ylim(0.4 * min(sp_target.min(), sp_nll.min(), sp_grad.min()), 2 * max(sp_target.max(), sp_grad.max()))
    else:
        ax.text(0.5, 0.5, "needs both backends", ha="center", transform=ax.transAxes)
    ax.set_xlabel("number of antennas"); ax.set_ylabel("CPU time / GPU time")
    ax.set_title(f"Speedup (CPU on {threads} thread{'s' if threads != 1 else ''})")

    for ax in axes:
        for r in rows:
            if r["layout"] == "chord":
                ax.axvline(r["n_ant"], color="0.5", ls=":", lw=1)
                ax.text(r["n_ant"], 1.0, " CHORD", transform=ax.get_xaxis_transform(),
                        va="top", ha="left", fontsize=8, color="0.4")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which="major", alpha=0.3)
#  SNR {cfg.get('snr', '?'):g}, 
    fig.suptitle(f"CorrCal calibration benchmark:"
                 f"{cfg.get('n_src_cal', '?')} sources in Σ, target = noise floor + {100 * cfg.get('target_tol', 0):.0f}%")
    fig.tight_layout()
    out = args.out or args.npz.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()