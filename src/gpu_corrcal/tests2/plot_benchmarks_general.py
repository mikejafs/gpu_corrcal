# ============================================================
# plot_benchmarks.py
# Loads saved .npz benchmark data and plots it.
# NO CuPy / GPU / corrcal imports — runs anywhere, instantly.
#
# Schema it reads (written by run_benchmarks._save_results):
#   sweep         : 'nant' | 'neig' | 'nsrc' | 'neig_nsrc'
#   x_values      : (n_points,)
#   series_names  : (n_series,)  str  — legend labels
#   series_times  : (n_series, n_points)  float — GPU times (s)
#   title_label / file_label + fixed params
#
# The plotter loops over whatever series it finds, so 2/3/4-line
# plots all use the same code path.
# ============================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

PLOT_DIR = "/home/mikej/main_phd_work/thesis_projects/gpu_corrcal/src/gpu_corrcal/tests2/benchmark_plots"

X_LABELS = {
    "nant":      r"$\mathbf{Number\ of\ Antennas}$",
    "neig":      r"$\mathbf{Number\ of\ Eigenmodes}$",
    "nsrc":      r"$\mathbf{Number\ of\ Sources}$",
    "neig_nsrc": r"$\mathbf{Number\ of\ Eigenmodes\ and\ Sources}$",
}

# marker cycle — extended past 3 so 4+ series each get a distinct marker
MARKERS = ["-P", "-p", "-o", "-s", "-^", "-D", "-v", "-X"]


# ------------------------------------------------------------
def load_benchmark(npz_path):
    """Load a benchmark npz into a plain dict (0-d scalars unwrapped)."""
    d = np.load(npz_path, allow_pickle=False)
    out = {}
    for k in d.files:
        arr = d[k]
        out[k] = arr.item() if arr.ndim == 0 else arr
    return out


def _iter_series(b):
    """Yield (label, times_array) for each series in a loaded benchmark.
    Handles both the new N-series schema and old 3-key files."""
    if "series_times" in b:
        names = [str(n) for n in b["series_names"]]
        for name, row in zip(names, b["series_times"]):
            yield name, row
    else:  # legacy files: test/reference/cpu triple
        for key, label in [("test_times", "Custom GPU Inverse Cov"),
                           ("reference_times", "CuPy Inverse Cov"),
                           ("cpu_reference_times", "CPU CorrCal Inverse Cov")]:
            if key in b:
                yield label, b[key]


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


def _save_fig(npz_path, save_plot, suffix=""):
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

    series : list of (label, times_array) pairs, same order as the
             data lines on `ax` (colors are matched by index).
    fixed_alpha : if given, skip the fit and use this exponent.
    min_t : if given, clip fit lines below this time.
    """
    fit_colors = [line.get_color() for line in ax.get_lines()[:len(series)]]

    for k, (lbl, times_arr) in enumerate(series):
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
    """Mark an interpolated point (e.g. CHORD's 512 antennas) on a curve."""
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
            f"(N_ant={n_ant}, {t_mark*1e6:.0f} us)",
            fontsize=10, color=color, va="center")


# ------------------------------------------------------------
def plot_benchmark(npz_path, save_plot=False, only=None, exclude=None,
                   power_law_fits=False, fit_on=None, chord_marker=False,
                   chord_series=0, **fit_kwargs):
    """Generic plotter — dispatches on the 'sweep' field, draws every series.

    only     : optional list of series labels to keep (others dropped)
    exclude  : optional list of series labels to drop
    fit_on   : optional list of labels to fit (default: all shown series)
    chord_series : index (into shown series) whose curve gets the 512 marker
    """
    b = load_benchmark(npz_path)
    sweep = str(b["sweep"])
    x = b["x_values"]

    # collect + filter series
    series = list(_iter_series(b))
    if only is not None:
        series = [(l, t) for l, t in series if l in only]
    if exclude is not None:
        series = [(l, t) for l, t in series if l not in exclude]

    _setup_style()
    fig, ax = plt.subplots()

    plot_fn = ax.loglog if sweep == "nant" else ax.semilogy
    for (lbl, times), m in zip(series, MARKERS):
        plot_fn(x, times, m, ms=9, label=lbl)

    if sweep == "nant" and power_law_fits:
        fit_series = series if fit_on is None else [(l, t) for l, t in series if l in fit_on]
        add_power_law_fits(ax, x, fit_series, **fit_kwargs)

    if sweep == "nant" and chord_marker and series:
        add_nant_marker(ax, x, series[chord_series][1], n_ant=512)

    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_major_formatter(FixedFormatter([rf"${int(v)}$" for v in x]))
    ax.tick_params(axis="both", which="major", labelsize=18, length=6, width=1.5)
    ax.set_xlabel(X_LABELS.get(sweep, ""), fontsize=18)
    ax.set_ylabel(r"$\mathbf{Average\ Run\ Time\ (s)}$", fontsize=18)
    ax.set_title(_make_title(b), fontsize=19)
    ax.grid(axis="y", alpha=0.5)
    ax.legend(fontsize=16)

    _save_fig(npz_path, save_plot)
    plt.show()
    return fig, ax


# ------------------------------------------------------------
def compare_benchmarks(npz_paths, series_label=None, labels=None,
                       save_plot=False):
    """Overlay ONE series from multiple runs (e.g. 5070 vs A40).

    series_label : which series to pull from each file (by its label).
                   If None, uses the first series in each file.
    labels       : optional per-file legend labels (default: file_label).
    """
    _setup_style()
    fig, ax = plt.subplots()

    sweep = None
    for i, path in enumerate(npz_paths):
        b = load_benchmark(path)
        sweep = str(b["sweep"])
        avail = dict(_iter_series(b))
        if series_label is not None and series_label in avail:
            times = avail[series_label]
        else:
            times = next(iter(avail.values()))  # first series
        lbl = labels[i] if labels else str(b.get("file_label", os.path.basename(path)))
        plot_fn = ax.loglog if sweep == "nant" else ax.semilogy
        plot_fn(b["x_values"], times, "-o", ms=8, label=lbl)

    ax.tick_params(axis="both", which="major", labelsize=18, length=6, width=1.5)
    ax.set_xlabel(X_LABELS.get(sweep, ""), fontsize=18)
    ax.set_ylabel(r"$\mathbf{Average\ Run\ Time\ (s)}$", fontsize=18)
    ax.grid(axis="y", alpha=0.5)
    ax.legend(fontsize=16)

    if save_plot:
        os.makedirs(PLOT_DIR, exist_ok=True)
        tag = (series_label or "first").replace(" ", "_")
        out = f"{PLOT_DIR}/comparison_{tag}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {out}")
    plt.show()
    return fig, ax


# =========================================================================
if __name__ == "__main__":
    # One file, all its series (2, 3, or 4 lines — whatever the target had):
    # file_name = "src/gpu_corrcal/tests2/benchmark_data/nant_targetlogdet_vs_onlyinv_n_eig3_n_src5_tpb128_seed42_5070_20260730_183002.npz"
    file_name = "src/gpu_corrcal/tests2/benchmark_data/nant_targetlogdet_vs_onlyinv_n_eig3_n_src5_tpb128_seed42_5070_20260803_173801.npz"
    # plot_benchmark(file_name)

    # With fits, only fit the CPU line, mark CHORD's 512 antennas:
    plot_benchmark(file_name, power_law_fits=True, fit_on=["Inv+logdet"],
                   chord_marker=True, min_t = 1e-5, save_plot=True)

    # Drop a series without regenerating data:
    # plot_benchmark(path, exclude=["CPU"])

    # Compare the same series across machines:
    # compare_benchmarks(["..._5070_....npz", "..._A40_....npz"],
    #                    series_label="Custom GPU")
    pass