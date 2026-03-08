import argparse
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Set, Callable

import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa

from matplotlib.ticker import LogLocator, FuncFormatter, NullLocator, NullFormatter

from adaptive_filter.initial_filter_params import InitialFilterParams
from adaptive_filter.tools import resolve_phoneme_set
from adaptive_filter.constants import IPA_CLASSES
from adaptive_filter.trapezoid_filtering import (
    trapezoid_filter_fft,
    _get_trapezoid_filterbank,
)


def format_frequency_major(x, pos):
    if x >= 1000:
        return "{:d}k".format(int(x / 1000))
    return "{:d}".format(int(x))


def format_frequency_minor(x, pos):
    if x < 20:
        return ""
    if x >= 1000:
        return "{:d}k".format(int(x / 1000))
    return "{:d}".format(int(x))


def configure_frequency_axis(ax, use_log_scale: bool) -> None:
    """
    Configure x-axis scale and tick formatting.
    """
    if use_log_scale:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_major_formatter(FuncFormatter(format_frequency_major))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="major", labelsize=11)
        ax.tick_params(axis="x", which="minor", labelsize=8, labelbottom=True)
    else:
        ax.set_xscale("linear")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())


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
    """
    return x_axis, spectrum


def no_overlay_provider(
    base_x_axis: np.ndarray,
) -> List[dict]:
    """
    Default overlay provider: return no overlays.
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
    spectrum_transform_fn: Optional[
        Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    ] = None,
    overlay_provider_fn: Optional[Callable[[np.ndarray], List[dict]]] = None,
    eps: float = 1e-11,
    x_ticks_from_overlay=True,
):
    """
    Generic plotting function for phoneme/spectrum dictionaries.
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
        if x_ticks_from_overlay:
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

    if all_x_ticks:
        tick_values = np.unique(np.concatenate(all_x_ticks).astype(np.float64))
        ax.set_xticks(tick_values)

        def _format_tick(x):
            if abs(x) < 100:
                return "{:.1f}".format(x)
            return "{:d}".format(int(round(x)))

        tick_labels = [_format_tick(x) for x in tick_values]

        ax.set_xticklabels(
            tick_labels,
            rotation=45,
            ha="right",
            fontsize=6,
        )

    ax.minorticks_off()
    ax.grid(True, which="major", axis="x", alpha=0.45, linewidth=0.6)

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
    """
    projection_matrix = np.asarray(projection_matrix)
    new_x_axis = np.asarray(new_x_axis)

    def transform_fn(spectrum: np.ndarray, x_axis: np.ndarray):
        del x_axis
        projected_values = projection_matrix @ spectrum
        return new_x_axis, projected_values

    return transform_fn


# ==========================================================
# Overlay providers
# ==========================================================

def build_filterbank_overlay_provider(
    filterbank_matrix: np.ndarray,
    filterbank_x_axis: np.ndarray,
    max_filters_to_plot: Optional[int] = None,
    alpha: float = 0.35,
):
    """
    Build an overlay provider for plotting filterbank curves.
    """
    filterbank_matrix = np.asarray(filterbank_matrix)
    filterbank_x_axis = np.asarray(filterbank_x_axis)

    def overlay_provider_fn(base_x_axis: np.ndarray):
        del base_x_axis
        overlays = []
        n_filters_local = filterbank_matrix.shape[0]
        n_to_plot = (
            n_filters_local
            if max_filters_to_plot is None
            else min(n_filters_local, max_filters_to_plot)
        )

        for filter_index in range(n_to_plot):
            overlays.append(
                {
                    "x": filterbank_x_axis,
                    "y": 10.0 * filterbank_matrix[filter_index],
                    "label": "filter_{}".format(filter_index) if n_to_plot <= 12 else None,
                    "color": "black",
                    "linestyle": "-",
                    "linewidth": 1.0,
                    "alpha": alpha,
                }
            )
        return overlays

    return overlay_provider_fn


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
        del base_x_axis
        overlays = []
        for x in marker_x_values:
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


def build_mel_filterbank_overlay_provider(
    mel_filterbank: np.ndarray,
    fft_freq_axis_hz: np.ndarray,
    max_filters_to_plot: Optional[int] = None,
    alpha: float = 0.35,
):
    """
    Build an overlay provider for plotting mel filter shapes.
    """
    mel_filterbank = np.asarray(mel_filterbank, dtype=np.float64)
    fft_freq_axis_hz = np.asarray(fft_freq_axis_hz, dtype=np.float64)

    def overlay_provider_fn(base_x_axis):
        del base_x_axis
        overlays = []
        n_filters_local = mel_filterbank.shape[0]
        n_to_plot = (
            n_filters_local
            if max_filters_to_plot is None
            else min(n_filters_local, max_filters_to_plot)
        )

        for filter_index in range(n_to_plot):
            overlays.append(
                {
                    "x": fft_freq_axis_hz,
                    "y": mel_filterbank[filter_index]*1000,
                    "label": "mel_{:03d}".format(filter_index) if n_to_plot <= 12 else None,
                    "color": "black",
                    "linestyle": "-",
                    "linewidth": 1.0,
                    "alpha": alpha,
                }
            )
        return overlays

    return overlay_provider_fn


# ==========================================================
# Data loading
# ==========================================================

def load_phoneme_amp_dict(pkl_path: Path) -> dict:
    """
    Load dict of average spectrum amplitude per phoneme.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


# ==========================================================
# Trapezoid projection transform
# ==========================================================

def build_trapezoid_projection_transform(
    edges,
    band_centers_hz: np.ndarray,
    overlap: float = 0.25,
):
    """
    Build a transform callback for plot_spectrum_dict_template that projects
    an FFT spectrum into trapezoid-filterbank band values.

    Parameters
    ----------
    edges : array-like
        Filter edge indices for `trapezoid_filter_fft`.
    band_centers_hz : np.ndarray
        X-axis to use for plotted trapezoid band values.
    overlap : float
        Trapezoid side-ramp fraction passed to `trapezoid_filter_fft`.
    apply_log : bool
        If True, convert band values to dB inside the transform.
    eps : float
        Numerical floor for log conversion.
    """
    edges = np.asarray(edges, dtype=np.int64)
    band_centers_hz = np.asarray(band_centers_hz, dtype=np.float64)

    def transform_fn(values: np.ndarray, x_axis: np.ndarray):
        del x_axis
        fft_tensor = torch.as_tensor(values, dtype=torch.float32)

        band_values = trapezoid_filter_fft(
            fft_input=fft_tensor,
            edges=edges,
            overlap=overlap,
        )

        band_values_np = band_values.detach().cpu().numpy()

        return band_centers_hz, band_values_np

    return transform_fn


def build_mel_projection_transform(
    mel_filterbank: np.ndarray,
    mel_centers_hz: np.ndarray,
):
    """
    Build a transform callback that projects FFT spectra into mel-band energies
    and returns mel center frequencies.
    """
    mel_filterbank = np.asarray(mel_filterbank, dtype=np.float64)
    mel_centers_hz = np.asarray(mel_centers_hz, dtype=np.float64)

    def transform_fn(values, x_axis):
        del x_axis
        values = np.asarray(values, dtype=np.float64)
        mel_energy = mel_filterbank @ (values ** 2)
        return mel_centers_hz, mel_energy

    return transform_fn


def compute_mel_filterbank_and_centers(
    sr: int,
    n_mels: int,
    n_fft: int,
    fmax: float,
):
    """
    Return mel filterbank matrix and its center frequencies in Hz.
    """
    mel_filterbank = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0,
        fmax=fmax,
        htk=True,
    )

    fft_freq_axis_hz = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    mel_centers_hz = np.zeros(n_mels, dtype=np.float64)

    for filter_index, mel_filter in enumerate(mel_filterbank):
        weight_sum = np.sum(mel_filter)
        if weight_sum > 0.0:
            mel_centers_hz[filter_index] = np.sum(mel_filter * fft_freq_axis_hz) / weight_sum

    return mel_filterbank, mel_centers_hz


# ==========================================================
# Plot helpers
# ==========================================================

def save_plot(
    output_path: Path,
    spectra_by_label: Dict[str, np.ndarray],
    x_axis: np.ndarray,
    selected_labels: Set[str],
    title: str,
    sample_rate: int,
    convert_to_db: bool,
    spectrum_transform_fn=None,
    overlay_provider_fn=None,
    x_ticks_from_overlay=True
) -> None:
    fig, _ = plot_spectrum_dict_template(
        spectra_by_label=spectra_by_label,
        x_axis=x_axis,
        selected_labels=selected_labels,
        title=title,
        sample_rate=sample_rate,
        n_total_bins=len(x_axis),
        convert_to_db=convert_to_db,
        show_legend=False,
        show_vertical_freq_lines=False,
        use_log_frequency_axis=False,
        spectrum_transform_fn=spectrum_transform_fn,
        overlay_provider_fn=overlay_provider_fn,
        x_ticks_from_overlay=x_ticks_from_overlay
    )
    fig.savefig(output_path)
    plt.close(fig)


def get_selected_phonemes() -> Set[str]:
    vowel_phonemes = resolve_phoneme_set(IPA_CLASSES, ["vowels"])
    consonant_phonemes = resolve_phoneme_set(IPA_CLASSES, ["consonants"])
    return vowel_phonemes | consonant_phonemes


def fit_adaptive_trapezoid_edges(
    phoneme_fft: Dict[str, np.ndarray],
    n_filters: int,
    bin_start: int,
    bin_end: int,
    contrast_weight: float,
    variance_weight: float,
    rel_variance_weight: float,
    energy_weight: float,
) -> np.ndarray:
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
    return np.asarray(initial_filter_params.edges, dtype=np.int64)


def compute_trapezoid_plot_artifacts(
    sr: int,
    n_fft_bins: int,
    edges: np.ndarray,
    overlap: float,
    n_filters: int,
):
    fft_freq_axis_hz = np.linspace(1.0, sr / 2.0, n_fft_bins)

    filter_left_hz = fft_freq_axis_hz[np.clip(edges[:-1], 0, n_fft_bins - 1)]
    filter_right_hz = fft_freq_axis_hz[np.clip(edges[1:] - 1, 0, n_fft_bins - 1)]
    filter_center_hz = 0.5 * (filter_left_hz + filter_right_hz)

    n_fft_for_filterbank = 2 * (n_fft_bins - 1)
    trapezoid_filterbank = _get_trapezoid_filterbank(
        n_fft=n_fft_for_filterbank,
        edges=edges,
        overlap=overlap,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).detach().cpu().numpy()

    trapezoid_transform_fn = build_trapezoid_projection_transform(
        edges=edges,
        band_centers_hz=filter_center_hz,
        overlap=overlap,
    )

    trapezoid_overlay_fn = build_filterbank_overlay_provider(
        filterbank_matrix=trapezoid_filterbank,
        filterbank_x_axis=fft_freq_axis_hz,
        max_filters_to_plot=n_filters,
        alpha=0.30,
    )

    return fft_freq_axis_hz, filter_center_hz, trapezoid_transform_fn, trapezoid_overlay_fn


def save_mel_projection_plot(
    output_path: Path,
    phoneme_fft: Dict[str, np.ndarray],
    phonemes: Set[str],
    sr: int,
    n_mels: int,
    n_fft: int,
    fmax: float,
    title: str,
) -> None:
    mel_filterbank, mel_centers_hz = compute_mel_filterbank_and_centers(
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        fmax=fmax,
    )

    mel_transform_fn = build_mel_projection_transform(
        mel_filterbank=mel_filterbank,
        mel_centers_hz=mel_centers_hz,
    )

    fft_freq_axis_hz = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)

    mel_overlay_fn = build_mel_filterbank_overlay_provider(
        mel_filterbank=mel_filterbank,
        fft_freq_axis_hz=fft_freq_axis_hz,
        max_filters_to_plot=n_mels,
        alpha=0.25,
    )

    save_plot(
        output_path=output_path,
        spectra_by_label=phoneme_fft,
        x_axis=fft_freq_axis_hz,
        selected_labels=phonemes,
        title=title,
        sample_rate=sr,
        convert_to_db=True,
        spectrum_transform_fn=mel_transform_fn,
        overlay_provider_fn=mel_overlay_fn,
        x_ticks_from_overlay=False
    )


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

    phoneme_fft = load_phoneme_amp_dict(phoneme_energy_pkl)
    phonemes = get_selected_phonemes()

    edges = fit_adaptive_trapezoid_edges(
        phoneme_fft=phoneme_fft,
        n_filters=n_filters,
        bin_start=bin_start,
        bin_end=bin_end,
        contrast_weight=contrast_weight,
        variance_weight=variance_weight,
        rel_variance_weight=rel_variance_weight,
        energy_weight=energy_weight,
    )

    n_fft_bins = len(next(iter(phoneme_fft.values())))
    (
        fft_freq_axis_hz,
        filter_center_hz,
        trapezoid_transform_fn,
        trapezoid_overlay_fn,
    ) = compute_trapezoid_plot_artifacts(
        sr=sr,
        n_fft_bins=n_fft_bins,
        edges=edges,
        overlap=0.25,
        n_filters=n_filters,
    )

    # 1. Original FFT spectra
    save_plot(
        output_path=output_dir / "original_phoneme_fft.jpg",
        spectra_by_label=phoneme_fft,
        x_axis=fft_freq_axis_hz,
        selected_labels=phonemes,
        title="Original phoneme FFT spectra (dB)",
        sample_rate=sr,
        convert_to_db=True,
    )

    # 2. Trapezoid-filtered spectra + filter shapes
    save_plot(
        output_path=output_dir / "trapezoid_filtered_with_filters.jpg",
        spectra_by_label=phoneme_fft,
        x_axis=fft_freq_axis_hz,
        selected_labels=phonemes,
        title="Trapezoid-filtered phoneme spectra with filter shapes",
        sample_rate=sr,
        convert_to_db=True,
        spectrum_transform_fn=trapezoid_transform_fn,
        overlay_provider_fn=trapezoid_overlay_fn,
        # x_ticks_from_overlay=False
    )

    # 3. 128 mel filters, 0 - 11025 Hz
    save_mel_projection_plot(
        output_path=output_dir / "phoneme_mel_128_11025hz_with_filters.jpg",
        phoneme_fft=phoneme_fft,
        phonemes=phonemes,
        sr=22050,
        n_mels=128,
        n_fft=1022,
        fmax=11025.0,
        title="Phoneme FFT projected to 128 mel bands + mel filter shapes",
    )

    # 4. 80 mel filters, 0 - 8000 Hz
    save_mel_projection_plot(
        output_path=output_dir / "phoneme_mel_80_8000hz_with_filters.jpg",
        phoneme_fft=phoneme_fft,
        phonemes=phonemes,
        sr=16000,
        n_mels=80,
        n_fft=1022,
        fmax=8000.0,
        title="Phoneme FFT projected to 80 mel bands + mel filter shapes",
    )


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