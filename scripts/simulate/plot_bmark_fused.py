"""
Plot the output of calibration_benchmark_fixedtiming.py for runs that include
the fused GPU backend (gpu_fused).

For older .npz files (backends gpu, cpu only, no t_point), use plot_bmark.py.

python plot_bmark_fused.py benchmark_calibration_fused.npz
python plot_bmark_fused.py benchmark_calibration_fused.npz -o fig.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

STYLE = {"cpu": dict(color="C0", marker="o", label="CPU (corrcal)"),
         "gpu": dict(color="C1", marker="s", label="GPU nll + grad"),
         "gpu_fused": dict(color="C2", marker="D", label="GPU fused")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", type=Path, default=Path("benchmark_calibration_fused.npz"))
    ap.add_argument("-o", "--out", type=Path, default=None, help="output image (default: <npz stem>.png)")
    args = ap.parse_args()

    f = np.load(args.npz, allow_pickle=True)
    rows = sorted(f["rows"], key=lambda r: r["n_ant"])
    backends = [str(b) for b in f["backends"]]
    cfg = f["config"].item()
    threads = cfg.get("cpu_threads", "?")

    if "t_point" not in "".join(rows[0].keys()):
        raise SystemExit(f"{args.npz} has no per-trial-point times (pre-fused run); plot it with plot_bmark.py")

    n_ant = np.array([r["n_ant"] for r in rows])
    get = lambda b, k: np.array([r[f"{b}_{k}"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) time to the common accuracy target
    ax = axes[0]
    for b in backends:
        ax.loglog(n_ant, get(b, "t_target"), ls="-", **STYLE[b])
    ax.set_xlabel("number of antennas"); ax.set_ylabel("time to target [s]")
    ax.set_title("Calibration time to noise-limited accuracy")

    # (b) cost of one scipy trial point: nll + grad, or one fused call
    ax = axes[1]
    for b in backends:
        st = dict(STYLE[b]); lab = st.pop("label")
        ax.loglog(n_ant, 1e3 * get(b, "t_point"), ls="-", label=f"{lab}: trial point", **st)
        if b != "gpu_fused":
            ax.loglog(n_ant, 1e3 * get(b, "t_nll"), ls=":", mfc="none", alpha=0.6, label=f"{lab}: nll", **st)
            ax.loglog(n_ant, 1e3 * get(b, "t_grad"), ls="--", mfc="none", alpha=0.6, label=f"{lab}: grad", **st)
    ax.set_xlabel("number of antennas"); ax.set_ylabel("time [ms]")
    ax.set_title("Objective and gradient, one trial point")

    # (c) speedup over CPU, per GPU backend
    ax = axes[2]
    gpu_bs = [b for b in backends if b != "cpu"]
    if "cpu" in backends and gpu_bs:
        all_sp = []
        for b in gpu_bs:
            c, lab = STYLE[b]["color"], STYLE[b]["label"]
            sp_target = get("cpu", "t_target") / get(b, "t_target")
            sp_point = get("cpu", "t_point") / get(b, "t_point")
            all_sp += [sp_target, sp_point]
            ax.loglog(n_ant, sp_target, "-o", color=c, label=f"{lab}: time to target")
            ax.loglog(n_ant, sp_point, "--v", color=c, mfc="none", label=f"{lab}: trial point")
            for x, y in zip(n_ant, sp_target):
                ax.annotate(f"{y:.1f}×" if y < 10 else f"{y:.0f}×", (x, y), textcoords="offset points",
                            xytext=(-6, -14), ha="right", fontsize=9, color=c)
        ax.axhline(1, color="0.6", lw=0.8)
        ax.set_ylim(0.4 * min(s.min() for s in all_sp), 2 * max(s.max() for s in all_sp))
    else:
        ax.text(0.5, 0.5, "needs cpu and a GPU backend", ha="center", transform=ax.transAxes)
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

    fig.suptitle(f"CorrCal calibration benchmark: "
                 f"{cfg.get('n_src_cal', '?')} sources in Σ, target = noise floor + {100 * cfg.get('target_tol', 0):.0f}%")
    fig.tight_layout()
    out = args.out or args.npz.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()