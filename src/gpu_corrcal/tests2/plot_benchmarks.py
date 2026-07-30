# ============================================================
# plot_benchmarks.py
# Loads saved .npz benchmark data and plots it.
# NO CuPy / GPU / corrcal imports — runs anywhere, instantly.
# ============================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

PLOT_DIR = "/home/mikej/main_phd_work/thesis_projects/gpu_corrcal/src/gpu_corrcal/tests2/benchmark_data"

X_LABELS = {
    "nant":      r"$\mathbf{Number\ of\ Antennas}$",
    "neig":      r"$\mathbf{Number\ of\ Eigenmodes}$",
    "nsrc":      r"$\mathbf{Number\ of\ Sources}$",
    "neig_nsrc": r"$\mathbf{Number\ of\ Eigenmodes\ and\ Sources}$",
}


# ------------------------------------------------------------
def load_benchmark(npz_path):
    """Load a benchmark npz into a plain dict (scalars unwrapped)."""
    d = np.load(npz_path, allow_pickle=False)
    out = {}
    for k in d.files:
        arr = d[k]
        out[k] = arr.item() if arr.ndim == 0 else arr
    return out


def _setup_style(figsize=(13, 5)):
    plt.rcParams["text.usetex"] = False
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
    })


def _make_title(b):
    """Build a title from whatever fixed params the file contains."""
    parts = []
    if "n_ant" in b:
        parts.append(rf"{int(b['n_ant'])}\ Antennas")
    if "n_eig" in b:
        parts.append(rf"{int(b['n_eig'])}\ Eigenmodes")
    if "n_src" in b:
        parts.append(rf"{int(b['n_src'])}\ Sources")
    inner = ",\ ".join(parts)
    label = b.get("title_label", "")
    if label:
        return rf"$\mathbf{{{label}\ ({inner})}}$"
    return rf"$\mathbf{{{inner}}}$"


def _save_fig(b, npz_path, save_plot, suffix=""):
    if not save_plot:
        return
    os.makedirs(PLOT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(npz_path))[0]
    out = f"{PLOT_DIR}/{base}{suffix}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {out}")


# ------------------------------------------------------------
def add_power_law_fits(ax, x, series, n_ant_ref=1000, n_ant_cutoff=500,
                       fixed_alpha=None, min_t=None):
    """Overlay anchored power-law fits on a log-log plot.

    series : list of (times_array, label) pairs, in the same order
             the data lines were added to `ax` (colors are matched).
    fixed_alpha : if given, skip the fit and use this exponent.
    min_t : if given, clip fit lines below this time (avoids the fit
            line diving under the latency floor).
    """
    fit_colors = [line.get_color() for line in ax.get_lines()[:len(series)]]

    for k, (times_arr, lbl) in enumerate(series):
        mask = x >= n_ant_cutoff
        if mask.sum() < 2 and fixed_alpha is None:
            continue

        if fixed_alpha is not None:
            alpha = fixed_alpha
        else:
            alpha, _ = np.polyfit(np.log10(x[mask]), np.log10(times_arr[mask]), 1)

        idx_ref = np.argmin(np.abs(x - n_ant_ref))
        A = times_arr[idx_ref] / (n_ant_ref ** alpha)

        x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        t_fit = A * x_fit ** alpha
        if min_t is not None:
            keep = t_fit >= min_t
            x_fit, t_fit = x_fit[keep], t_fit[keep]

        ax.loglog(x_fit, t_fit, "--", color=fit_colors[k],
                  linewidth=2.5, alpha=0.55,
                  label=rf"{lbl} fit: $\propto N^{{{alpha:.2f}}}$")


def add_nant_marker(ax, x, times, n_ant=512, color="red"):
    """Mark an interpolated point (e.g. CHORD's 512 antennas) on a log-log curve."""
    from scipy.interpolate import interp1d
    f = interp1d(np.log10(x), np.log10(times))
    t_mark = 10 ** float(f(np.log10(n_ant)))

    ax.plot(n_ant, t_mark, "o", color=color, markersize=5, zorder=5)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot([n_ant, n_ant], [ylim[0], t_mark], "--", color=color, alpha=0.4, lw=1)
    ax.plot([xlim[0], n_ant], [t_mark, t_mark], "--", color=color, alpha=0.4, lw=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.text(n_ant * 1.3, t_mark,
            f"(N_ant={n_ant}, {t_mark*1e6:.0f} μs)",
            fontsize=10, color=color, va="center")


# ------------------------------------------------------------
def plot_benchmark(npz_path, save_plot=False, show_cpu=True,
                   power_law_fits=False, chord_marker=False, **fit_kwargs):
    """Generic plotter — dispatches on the 'sweep' field in the file."""
    b = load_benchmark(npz_path)
    sweep = b["sweep"]
    x = b["x_values"]

    _setup_style()
    fig, ax = plt.subplots()

    series = [
        (b["test_times"],      "Custom GPU Inverse Cov"),
        (b["reference_times"], "CuPy Inverse Cov"),
    ]
    if show_cpu:
        series.append((b["cpu_reference_times"], "CPU CorrCal Inverse Cov"))

    plot_fn = ax.loglog if sweep == "nant" else ax.semilogy
    markers = ["-P", "-p", "-p"]
    for (times, lbl), m in zip(series, markers):
        plot_fn(x, times, m, ms=9, label=lbl)

    if sweep == "nant" and power_law_fits:
        fit_series = [(t, l.split()[0]) for t, l in series]
        add_power_law_fits(ax, x, fit_series, **fit_kwargs)

    if sweep == "nant" and chord_marker:
        add_nant_marker(ax, x, b["test_times"], n_ant=512)

    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_major_formatter(FixedFormatter([rf"${int(v)}$" for v in x]))
    ax.tick_params(axis="both", which="major", labelsize=18, length=6, width=1.5)
    ax.set_xlabel(X_LABELS[sweep], fontsize=18)
    ax.set_ylabel(r"$\mathbf{Average\ Run\ Time\ (s)}$", fontsize=18)
    ax.set_title(_make_title(b), fontsize=19)
    ax.grid(axis="y", alpha=0.5)
    ax.legend(fontsize=16)

    _save_fig(b, npz_path, save_plot)
    plt.show()
    return fig, ax


# ------------------------------------------------------------
def compare_benchmarks(npz_paths, series_key="test_times", labels=None,
                       save_plot=False):
    """Overlay one series from multiple runs (e.g. 5070 vs A40).

    series_key : which array to pull from each file
                 ('test_times', 'reference_times', 'cpu_reference_times')
    """
    _setup_style()
    fig, ax = plt.subplots()

    sweep = None
    for i, path in enumerate(npz_paths):
        b = load_benchmark(path)
        sweep = b["sweep"]
        lbl = labels[i] if labels else b.get("file_label", os.path.basename(path))
        plot_fn = ax.loglog if sweep == "nant" else ax.semilogy
        plot_fn(b["x_values"], b[series_key], "-o", ms=8, label=lbl)

    ax.tick_params(axis="both", which="major", labelsize=18, length=6, width=1.5)
    ax.set_xlabel(X_LABELS.get(sweep, ""), fontsize=18)
    ax.set_ylabel(r"$\mathbf{Average\ Run\ Time\ (s)}$", fontsize=18)
    ax.grid(axis="y", alpha=0.5)
    ax.legend(fontsize=16)

    if save_plot:
        os.makedirs(PLOT_DIR, exist_ok=True)
        out = f"{PLOT_DIR}/comparison_{series_key}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {out}")
    plt.show()
    return fig, ax


# =========================================================================
if __name__ == "__main__":
    # Point at any saved run:
    plot_name = "src/gpu_corrcal/tests2/benchmark_data/nant_targetlogdet_vs_onlyinv_n_eig3_n_src5_tpb128_seed42_5070_20260730_183002.npz"
    plot_benchmark(plot_name, power_law_fits=True, save_plot=False)

    # Compare device vs cluster on the same axes:
    # compare_benchmarks([
    #     "benchmark_data/nant_..._5070_....npz",
    #     "benchmark_data/nant_..._A40_....npz",
    # ], series_key="test_times")
    pass