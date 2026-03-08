import argparse
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Set

import numpy as np
import matplotlib.pyplot as plt
import torch
from adaptive_filter.initial_filter_params import InitialFilterParams
from adaptive_filter.tools import resolve_phoneme_set
from adaptive_filter.constants import IPA_CLASSES
from adaptive_filter.trapezoid_filtering import trapezoid_filter_fft, _get_trapezoid_filterbank


from matplotlib.ticker import LogLocator, FuncFormatter



def format_frequency_major(x, pos):
    if x >= 1000:
        return f"{int(x / 1000)}k"
    return f"{int(x)}"


def format_frequency_minor(x, pos):
    if x < 20:
        return ""
    if x >= 1000:
        return f"{int(x / 1000)}k"
    return f"{int(x)}"


def configure_frequency_axis(ax, use_log_scale: bool) -> None:
    """
    Configure x-axis scale and tick formatting.
    """
    if use_log_scale:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_major_formatter(FuncFormatter(format_frequency_major))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(FuncFormatter(format_frequency_minor))
        ax.tick_params(axis="x", which="major", labelsize=11)
        ax.tick_params(axis="x", which="minor", labelsize=8, labelbottom=True)
    else:
        ax.set_xscale("linear")


def draw_vertical_frequency_lines(ax, x_values: np.ndarray) -> None:
    """
    Draw vertical guide lines at the given x positions.
    """
    for x in x_values:
        ax.axvline(
            x=x,
            color="green",
            linewidth=1.5,
            alpha=0.5,
            linestyle="--",
        )


def choose_legend_columns(n_items: int) -> int:
    if n_items <= 12:
        return 1
    if n_items <= 24:
        return 2
    return 3


# ==========================================================
# Plot transform / overlay callback contracts
# ==========================================================

def identity_spectrum_transform(
    spectrum: np.ndarray,
    x_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Default spectrum transform: return the original x/y unchanged.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (new_x_axis, new_values)
    """
    return x_axis, spectrum


def no_overlay_provider(
    base_x_axis: np.ndarray,
) -> List[dict]:
    """
    Default overlay provider: return no overlays.

    Returns
    -------
    List[dict]
        Empty List.
    """
    return []


# ==========================================================
# Main template plotting function
# ==========================================================
def plot_spectrum_dict_template(
    spectra_by_label: Dict[str, np.ndarray],
    x_axis: Optional[np.ndarray] = None,
    selected_labels: Optional[Set[str]] = None,
    title: str = "",
    sample_rate: int = 22050,
    n_total_bins: Optional[int] = 512,
    convert_to_db: bool = True,
    show_legend: bool = True,
    show_vertical_freq_lines: bool = False,
    use_log_frequency_axis: bool = False,
    dpi: int = 100,
    width_px: int = 10000,
    height_px: int = 1000,
    spectrum_transform_fn=None,
    overlay_provider_fn=None,
    eps: float = 10e-12,
):
    """
    Generic plotting function for phoneme/spectrum dictionaries.

    Supports three modes:
    1. plot spectra as-is
    2. plot transformed spectra using `spectrum_transform_fn`
    3. add extra overlays using `overlay_provider_fn`

    Parameters
    ----------
    spectra_by_label : Dict[str, np.ndarray]
        Mapping: label -> 1D spectrum values.
    x_axis : Optional[np.ndarray]
        Base x-axis for the original spectra. If None, a default FFT-like axis is built.
    selected_labels : Optional[Set[str]]
        If provided, only these labels are plotted.
    title : str
        Plot title.
    sample_rate : int
        Audio sample rate in Hz.
    n_total_bins : Optional[int]
        Total number of bins in the original representation.
    convert_to_db : bool
        Whether to convert plotted spectrum values to dB.
    show_legend : bool
        Whether to show the legend.
    show_vertical_freq_lines : bool
        Whether to draw vertical guide lines on the base x-axis.
    use_log_frequency_axis : bool
        Whether to use log scale on the x-axis.
    dpi : int
        Figure DPI.
    width_px : int
        Figure width in pixels.
    height_px : int
        Figure height in pixels.
    spectrum_transform_fn : Optional[callable]
        Callback with signature:
            (values: np.ndarray, x_axis: np.ndarray) -> (new_x_axis, new_values)

        This can change both x and y, e.g. FFT bins -> mel bands.
    overlay_provider_fn : Optional[callable]
        Callback with signature:
            (base_x_axis: np.ndarray) -> List[dict]

        Each overlay dict can contain:
            {
                "x": np.ndarray,
                "y": np.ndarray,
                "label": Optional[str],
                "color": Optional[str],
                "linestyle": Optional[str],
                "linewidth": Optional[float],
                "alpha": Optional[float],
            }

        Example use: plot a filterbank curve or band markers.
    eps : float
        Stability epsilon for dB conversion.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created figure and axes.
    """
    if spectrum_transform_fn is None:
        spectrum_transform_fn = identity_spectrum_transform
    if overlay_provider_fn is None:
        overlay_provider_fn = no_overlay_provider

    if x_axis is None:
        if n_total_bins is None:
            first_values = next(iter(spectra_by_label.values()))
            n_total_bins = len(first_values)
        x_axis = np.linspace(0, sample_rate / 2, n_total_bins)
    else:
        x_axis = np.asarray(x_axis)

    fig, ax = plt.subplots(
        figsize=(width_px / dpi, height_px / dpi),
        dpi=dpi,
    )

    colors = list(plt.cm.tab20.colors)
    line_styles = ["-", "--", ":", "-."]

    color_index = 0
    line_style_index = 0
    plotted_labels = []
    all_x_ticks = []

    last_transformed_x = x_axis

    for label, values in sorted(spectra_by_label.items()):
        if selected_labels is not None and label not in selected_labels:
            continue

        values = np.asarray(values)

        transformed_x, transformed_y = spectrum_transform_fn(values, x_axis)
        transformed_x = np.asarray(transformed_x)
        transformed_y = np.asarray(transformed_y)

        last_transformed_x = transformed_x
        all_x_ticks.append(transformed_x)

        if convert_to_db:
            transformed_y = 20.0 * np.log10(transformed_y + eps)

        color = colors[color_index]
        line_style = line_styles[line_style_index]

        color_index += 1
        if color_index >= len(colors):
            color_index = 0
            line_style_index = (line_style_index + 1) % len(line_styles)

        ax.plot(
            transformed_x,
            transformed_y,
            label=label,
            color=color,
            linestyle=line_style,
            linewidth=2.0,
            alpha=0.9,
        )
        plotted_labels.append(label)

    if show_vertical_freq_lines:
        draw_vertical_frequency_lines(ax, last_transformed_x)

    overlays = overlay_provider_fn(last_transformed_x)
    for overlay in overlays:
        overlay_x = np.asarray(overlay["x"])
        all_x_ticks.append(overlay_x)

        ax.plot(
            overlay_x,
            overlay["y"],
            label=overlay.get("label"),
            color=overlay.get("color", "black"),
            linestyle=overlay.get("linestyle", "-"),
            linewidth=overlay.get("linewidth", 1.5),
            alpha=overlay.get("alpha", 0.8),
        )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)" if convert_to_db else "Magnitude")
    ax.set_title(title or "Spectra")

    configure_frequency_axis(ax, use_log_scale=use_log_frequency_axis)

    # --------------------------------------------------
    # X ticks: show every x value without scientific notation
    # --------------------------------------------------
    if all_x_ticks:
        tick_values = np.unique(np.concatenate(all_x_ticks).astype(np.float64))
        ax.set_xticks(tick_values)

        def _format_tick(x):
            if abs(x) < 100:
                return "{:.1f}".format(x)     # one decimal
            else:
                return "{:d}".format(int(round(x)))  # full digits

        tick_labels = [_format_tick(x) for x in tick_values]

        ax.set_xticklabels(
            tick_labels,
            rotation=45,
            ha="right",
            fontsize=6,  # one size smaller
        )

    # Stronger visible grid
    ax.grid(True, which="major", axis="both", alpha=0.45, linewidth=0.6)
    ax.grid(True, which="minor", axis="both", alpha=0.20, linewidth=0.4)

    if show_legend:
        n_legend_items = len(plotted_labels) + sum(
            1 for overlay in overlays if overlay.get("label")
        )
        ncol = choose_legend_columns(n_legend_items)
        ax.legend(frameon=False, fontsize=11, ncol=ncol)

    fig.tight_layout()
    return fig, ax

def build_matrix_projection_transform(
    projection_matrix: np.ndarray,
    new_x_axis: np.ndarray,
):
    """
    Build a transform callback that projects each spectrum with a matrix.

    Example:
        FFT(512,) -> mel(80,)
        projection_matrix shape = (80, 512)
        new_x_axis shape = (80,)

    Returns
    -------
    callable
        Function with signature:
            (label, spectrum, x_axis) -> (new_x_axis, projected_values)
    """
    projection_matrix = np.asarray(projection_matrix)
    new_x_axis = np.asarray(new_x_axis)

    def transform_fn(label: str, spectrum: np.ndarray, x_axis: np.ndarray):
        projected_values = projection_matrix @ spectrum
        return new_x_axis, projected_values

    return transform_fn


# ==========================================================
# Example overlay provider: full filterbank
# ==========================================================

def build_filterbank_overlay_provider(
    filterbank_matrix: np.ndarray,
    filterbank_x_axis: np.ndarray,
    max_filters_to_plot: Optional[ int ] = None,
    alpha: float = 0.35,
):
    """
    Build an overlay provider for plotting filterbank curves.

    Parameters
    ----------
    filterbank_matrix : np.ndarray
        Shape (n_filters, n_bins)
    filterbank_x_axis : np.ndarray
        X-axis for the filterbank curves, usually FFT-bin frequencies in Hz.
    max_filters_to_plot : Optional[ int ]
        Limit how many filters are plotted.
    alpha : float
        Overlay transparency.

    Returns
    -------
    callable
        Function with signature:
            (base_x_axis) -> list[dict]
    """
    filterbank_matrix = np.asarray(filterbank_matrix)
    filterbank_x_axis = np.asarray(filterbank_x_axis)

    def overlay_provider_fn(base_x_axis: np.ndarray):
        overlays = []
        n_filters = filterbank_matrix.shape[0]
        n_to_plot = n_filters if max_filters_to_plot is None else min(n_filters, max_filters_to_plot)

        for filter_index in range(n_to_plot):
            overlays.append(
                {
                    "x": filterbank_x_axis,
                    "y": 10*filterbank_matrix[filter_index],
                    "label": f"filter_{filter_index}" if n_to_plot <= 12 else None,
                    "color": "black",
                    "linestyle": "-",
                    "linewidth": 1.0,
                    "alpha": alpha,
                }
            )
        return overlays

    return overlay_provider_fn

def load_phoneme_amp_dict(pkl_path: Path) -> dict:
    """
    Load dict of average spectrum amplitude per phoneme
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data

# ==========================================================
# Example overlay provider: vertical markers only
# ==========================================================

def build_vertical_marker_overlay_provider(
    marker_x_values: np.ndarray,
    y_min: float = -120.0,
    y_max: float = 10.0,
    color: str = "green",
    alpha: float = 0.35,
    linestyle: str = "--",
):
    """
    Build an overlay provider that draws vertical marker lines as plot overlays.
    """
    marker_x_values = np.asarray(marker_x_values)

    def overlay_provider_fn(base_x_axis: np.ndarray):
        overlays = []
        for idx, x in enumerate(marker_x_values):
            overlays.append(
                {
                    "x": np.array([x, x]),
                    "y": np.array([y_min, y_max]),
                    "label": None,
                    "color": color,
                    "linestyle": linestyle,
                    "linewidth": 1.0,
                    "alpha": alpha,
                }
            )
        return overlays

    return overlay_provider_fn

def apply_trapezoid_filter_visualization(
    phoneme_fft_dict: Dict[str, np.ndarray],
    edges,
    phonemes=None,
    overlap: float = 0.25,
    log: bool = True,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Apply the trapezoid filterbank to a dictionary of FFT spectra.

    Parameters
    ----------
    phoneme_fft_dict : Dict[str, np.ndarray]
        Mapping from phoneme label to 1D FFT spectrum in linear domain.
    edges : array-like
        Filter edge indices for `trapezoid_filter_fft`.
    phonemes : Optional[ set[str] ]
        If provided, only these phonemes are processed.
    overlap : float, default=0.25
        Trapezoid side-ramp fraction passed to `trapezoid_filter_fft`.
    log : bool, default=True
        If True, convert filtered values to dB.
    eps : float, default=1e-12
        Numerical floor for log conversion.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from phoneme label to filtered band values.
    """
    filtered_by_phoneme: Dict[str, np.ndarray] = {}

    for phoneme, fft_spectrum in sorted(phoneme_fft_dict.items()):
        if phonemes is not None and phoneme not in phonemes:
            continue

        fft_tensor = torch.as_tensor(fft_spectrum, dtype=torch.float32)

        band_values = trapezoid_filter_fft(
            fft_input=fft_tensor,
            edges=edges,
            overlap=overlap,
        )

        band_values_np = band_values.detach().cpu().numpy()

        if log:
            band_values_np = 10.0 * np.log10(np.maximum(band_values_np, eps))

        filtered_by_phoneme[phoneme] = band_values_np

    return filtered_by_phoneme


def main(
    phoneme_energy_pkl: Path,
    sr: int,
    n_filters: int,
    bin_start: int,
    bin_end: int,
    contrast_weight: float,
    variance_weight: float,
    rel_variance_weight: float,
    energy_weight: float,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load phoneme FFT data
    # --------------------------------------------------
    phoneme_fft = load_phoneme_amp_dict(phoneme_energy_pkl)

    vowel_phonemes = resolve_phoneme_set(IPA_CLASSES, ["vowels"])
    consonant_phonemes = resolve_phoneme_set(IPA_CLASSES, ["consonants"])
    phonemes = vowel_phonemes | consonant_phonemes

    # --------------------------------------------------
    # Fit adaptive trapezoid filterbank
    # --------------------------------------------------
    initial_filter_params = InitialFilterParams(
        n_filters=n_filters,
        contrast_weight=contrast_weight,
        variance_weight=variance_weight,
        rel_variance_weight=rel_variance_weight,
        energy_weight=energy_weight,
        bin_start=bin_start,
        bin_end=bin_end,
    )
    initial_filter_params.fit(phoneme_fft)
    edges = np.asarray(initial_filter_params.edges, dtype=np.int64)

    # --------------------------------------------------
    # Apply trapezoid filterbank to phoneme FFT dict
    # --------------------------------------------------
    filtered_phoneme_fft = apply_trapezoid_filter_visualization(
        phoneme_fft_dict=phoneme_fft,
        edges=edges,
        phonemes=phonemes,
        overlap=0.25,
        log=True,
    )

    # --------------------------------------------------
    # Plot axes
    # --------------------------------------------------
    n_fft_bins = len(next(iter(phoneme_fft.values())))
    fft_freq_axis_hz = np.linspace(1.0, sr / 2, n_fft_bins)

    filter_left_hz = fft_freq_axis_hz[np.clip(edges[:-1], 0, n_fft_bins - 1)]
    filter_right_hz = fft_freq_axis_hz[np.clip(edges[1:] - 1, 0, n_fft_bins - 1)]
    filter_center_hz = 0.5 * (filter_left_hz + filter_right_hz)

    # --------------------------------------------------
    # Build trapezoid filter shapes for overlay
    # --------------------------------------------------
    n_fft_for_filterbank = 2 * (n_fft_bins - 1)
    trapezoid_filterbank = _get_trapezoid_filterbank(
        n_fft=n_fft_for_filterbank,
        edges=edges,
        overlap=0.25,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).detach().cpu().numpy()

    trapezoid_overlay_fn = build_filterbank_overlay_provider(
        filterbank_matrix=trapezoid_filterbank,
        filterbank_x_axis=fft_freq_axis_hz,
        max_filters_to_plot=n_filters,
        alpha=0.30,
    )

    # --------------------------------------------------
    # 1. Original FFT spectra
    # --------------------------------------------------
    fig, _ = plot_spectrum_dict_template(
        spectra_by_label=phoneme_fft,
        x_axis=fft_freq_axis_hz,
        selected_labels=phonemes,
        title="Original phoneme FFT spectra (dB)",
        sample_rate=sr,
        n_total_bins=n_fft_bins,
        convert_to_db=True,
        show_legend=False,
        show_vertical_freq_lines=False,
        use_log_frequency_axis=False,
    )
    fig.savefig(output_dir / "original_phoneme_fft.jpg")
    plt.close(fig)

    # --------------------------------------------------
    # 2. Trapezoid-filtered spectra
    # --------------------------------------------------
    fig, _ = plot_spectrum_dict_template(
        spectra_by_label=filtered_phoneme_fft,
        x_axis=filter_center_hz,
        selected_labels=phonemes,
        title="Trapezoid-filtered phoneme spectra (dB)",
        sample_rate=sr,
        n_total_bins=len(filter_center_hz),
        convert_to_db=False,  # already converted in apply_trapezoid_filter_visualization
        show_legend=False,
        show_vertical_freq_lines=False,
        use_log_frequency_axis=False,
    )
    fig.savefig(output_dir / "trapezoid_filtered_phoneme_fft.jpg")
    plt.close(fig)

    # --------------------------------------------------
    # 3. Trapezoid-filtered spectra + filter shapes
    # --------------------------------------------------
    fig, _ = plot_spectrum_dict_template(
        spectra_by_label=filtered_phoneme_fft,
        x_axis=filter_center_hz,
        selected_labels=phonemes,
        title="Trapezoid-filtered phoneme spectra with filter shapes",
        sample_rate=sr,
        n_total_bins=len(filter_center_hz),
        convert_to_db=False,  # already converted in apply_trapezoid_filter_visualization
        show_legend=False,
        show_vertical_freq_lines=False,
        use_log_frequency_axis=False,
        overlay_provider_fn=trapezoid_overlay_fn,
    )
    fig.savefig(output_dir / "trapezoid_filtered_with_filters.jpg")
    plt.close(fig)





 
# ============================================================
# Argparse
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize adaptive IIR filterbank on FFT-domain phoneme data"
    )

    parser.add_argument("--phoneme-energy-pkl", type=Path, required=True)
    parser.add_argument("--sr", type=int, default=22050)

    parser.add_argument("--n-filters", type=int, default=128)
    parser.add_argument("--bin-start", type=int, default=1)
    parser.add_argument("--bin-end", type=int, default=480)

    parser.add_argument("--contrast-weight", type=float, default=0.2)
    parser.add_argument("--variance-weight", type=float, default=0.2)
    parser.add_argument("--rel-variance-weight", type=float, default=0.1)
    parser.add_argument("--energy-weight", type=float, default=0.9)

    parser.add_argument("--output-dir", type=Path, default=Path("figures"))

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        phoneme_energy_pkl=args.phoneme_energy_pkl,
        sr=args.sr,
        n_filters=args.n_filters,
        bin_start=args.bin_start,
        bin_end=args.bin_end,
        contrast_weight=args.contrast_weight,
        variance_weight=args.variance_weight,
        rel_variance_weight=args.rel_variance_weight,
        energy_weight=args.energy_weight,
        output_dir=args.output_dir,
    )
