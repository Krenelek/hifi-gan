


import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

def normalize_minmax(x, eps=1e-8):
    x = x - x.min()
    return x / (x.max() + eps)


def plot_phoneme_spectra(
    spec_phoneme,
    freqs=None,
    phonemes=None,
    title="",
    sr=22050,
    bins=512,
    bins_total=512,
    log = True,
    legend = True,
    plot_freq_lines = False,
    x_log_scale = False,
    dpi = 100,
    width_px = 10000,
    height_px = 1000,
):
    eps = 1e-10
    if freqs is None:
        freqs = np.linspace(1, sr / 2, bins_total)[:bins]
    else:
        bins_total = len(freqs)


    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)

    colors = list(plt.cm.tab20.colors)  # 20 distinct colors
    linestyles = ["-", "--", ":", "-."]

    color_idx = 0
    linestyle_idx = 0

    for ph, spec in sorted(spec_phoneme.items()):
        if phonemes is not None and ph not in phonemes:
            continue

        spec = spec[:bins]

        color = colors[color_idx]
        linestyle = linestyles[linestyle_idx]

        color_idx += 1
        if color_idx >= len(colors):
            color_idx = 0
            linestyle_idx = (linestyle_idx + 1) % len(linestyles)

        if log:
            spec_db = 20 * np.log10(spec + eps) # dB = 20log(amp)
        else:
            spec_db = spec

        ax.plot(
            freqs,
            spec_db,
            label=ph,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            alpha=0.9,
        )
    # --------------------------------------------------
    # Vertical frequency lines
    # --------------------------------------------------
    if plot_freq_lines:
        for x in freqs:
            ax.axvline(
                x=x,
                color="green",
                linewidth=3.0,
                alpha=0.85,
                linestyle="--",
            )

    # --------------------------------------------------
    # Labels & title
    # --------------------------------------------------
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title or "Averaged phoneme spectra")
    ax.grid(True, alpha=0.3)

    # --------------------------------------------------
    # X scale & ticks
    # --------------------------------------------------
    if x_log_scale:
        ax.set_xscale("log")

        # ---------- MAJOR TICKS: 10^n ----------
        ax.xaxis.set_major_locator(LogLocator(base=10.0))

        def major_formatter(x, pos):
            if x >= 1000:
                return f"{int(x/1000)}k"
            return f"{int(x)}"

        ax.xaxis.set_major_formatter(FuncFormatter(major_formatter))

        # ---------- MINOR TICKS: 2..9 * 10^n ----------
        ax.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10))
        )

        def minor_formatter(x, pos):
            if x < 20:
                return ""
            if x >= 1000:
                return f"{int(x/1000)}k"
            return f"{int(x)}"

        ax.xaxis.set_minor_formatter(FuncFormatter(minor_formatter))

        # ---------- FORCE LABELS ----------
        ax.tick_params(axis="x", which="major", labelsize=11)
        ax.tick_params(axis="x", which="minor", labelsize=8, labelbottom=True)

    else:
        ax.set_xscale("linear")




    # --------------------------------------------------
    # Legend
    # --------------------------------------------------
    if legend:
        if phonemes is None or len(phonemes) <= 12:
            ax.legend(frameon=False, fontsize=11)
        else:
            ax.legend(frameon=False, ncol=2, fontsize=11)

    fig.tight_layout()



def resolve_phoneme_set(
    classes: dict,
    include: list,
) -> set:
    selected = set()

    for item in include:
        parts = item.split(".")
        node = classes
        for p in parts:
            node = node[p]

        if isinstance(node, set):
            selected |= node
        elif isinstance(node, dict):
            for v in node.values():
                selected |= v

    return selected



# def extract_band_energy_from_amp(filters):
#     """
#     x: 1D signal
#     returns: (128,) mean energy vector
#     """
#     energies = np.zeros(len(filters))

#     for i, XXX in enumerate(filters):
#         y = XXX
#         energies[i] = np.mean(y ** 2)

#     return energies


