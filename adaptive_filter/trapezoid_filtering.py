
import hashlib
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Global caches
# ============================================================

_trapezoid_basis_cache = {}
_hann_window_cache = {}


def _hash_edges(edges) -> str:
    edges_np = np.asarray(edges, dtype=np.int64).reshape(-1)
    return hashlib.sha1(edges_np.tobytes()).hexdigest()


def _get_hann_window(win_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (win_size, str(device), str(dtype))
    if key not in _hann_window_cache:
        _hann_window_cache[key] = torch.hann_window(
            win_size, device=device, dtype=dtype
        )
    return _hann_window_cache[key]


def _build_trapezoid_filterbank(
    n_fft: int,
    edges,
    overlap: float,
    slope_db: float = 60.0,
    device=None,
    dtype=torch.float32,
    normalize_energy: bool = True,
    eps=10e-12,
) -> torch.Tensor:
    """
    Build a trapezoid filterbank of shape (n_filters, n_fft//2 + 1).

    Parameters
    ----------
    n_fft : int
        FFT size.
    edges : array-like, shape (n_filters + 1,)
        Local integer bin edges.
    overlap : float
        Ramp width as a fraction of band width on each side.
        Recommended range: [0.0, 0.5].
    device : torch.device or str
        Output device.
    dtype : torch.dtype
        Output dtype.

    Returns
    -------
    fb : torch.Tensor
        Trapezoid bank, shape (n_filters, n_bins).
    """
    if overlap < 0:
        raise ValueError("overlap must be >= 0.0")

    n_bins = n_fft // 2 + 1

    edges = np.asarray(edges, dtype=np.int64).reshape(-1)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("edges must be a 1D array of length >= 2")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be strictly increasing")

    if edges[0] < 0 or edges[-1] > n_bins:
        raise ValueError(
            f"Shifted edges must stay inside [0, {n_bins}], got [{edges[0]}, {edges[-1]}]"
        )

    # Build in float32 for speed/stability, cast once at the end.
    work_dtype = torch.float32
    device = torch.device(device) if device is not None else torch.device("cpu")

    edges_t = torch.as_tensor(edges, device=device, dtype=work_dtype)
    starts = edges_t[:-1]          # (F,)
    ends = edges_t[1:]             # (F,)
    widths = ends - starts         # (F,)

    # Recommended for true trapezoids: overlap <= 0.5
    # We clamp to width/2 so the bank remains valid.
    ramps = torch.min(widths * float(overlap), widths * 0.5)  # (F,)

    # Bin coordinates
    x = torch.arange(n_bins, device=device, dtype=work_dtype).unsqueeze(0)  # (1, K)

    plateau_start = starts.unsqueeze(1)
    plateau_end = ends.unsqueeze(1)

    ramp_width = ramps.unsqueeze(1)

    support_start = plateau_start - ramp_width
    support_end = plateau_end + ramp_width

    # plateau mask
    plateau = (x >= plateau_start) & (x < plateau_end)

    # full support mask
    support = (x >= support_start) & (x < support_end)

    fb = torch.zeros((starts.shape[0], n_bins), device=device, dtype=work_dtype)

    # plateau = 1
    fb = torch.where(plateau, torch.ones_like(fb), fb)

    # Only compute ramps for filters that actually have ramp width > 0
    has_ramp = ramps > 0
    if torch.any(has_ramp):

        starts_r = starts[has_ramp].unsqueeze(1)
        ends_r = ends[has_ramp].unsqueeze(1)
        ramps_r = ramps[has_ramp].unsqueeze(1)

        plateau_start_r = starts_r
        plateau_end_r = ends_r

        support_start_r = plateau_start_r - ramps_r
        support_end_r = plateau_end_r + ramps_r

        x_r = x

        # rising ramp (support_start -> plateau_start)
        left_region = (x_r >= support_start_r) & (x_r < plateau_start_r)
        left_db = -slope_db + slope_db * (x_r - support_start_r) / ramps_r
            
        # falling ramp (plateau_end -> support_end)
        right_region = (x_r >= plateau_end_r) & (x_r < support_end_r)
        right_db = -slope_db + slope_db * (support_end_r - x_r) / ramps_r
            
        left_db = torch.clamp(left_db, -slope_db, 0.0)
        right_db = torch.clamp(right_db, -slope_db, 0.0)

        left_amp = torch.pow(10.0, left_db / 20.0)
        right_amp = torch.pow(10.0, right_db / 20.0)

        ramp_values = torch.zeros_like(left_amp)

        ramp_values = torch.where(left_region, left_amp, ramp_values)
        ramp_values = torch.where(right_region, right_amp, ramp_values)

        fb[has_ramp] = torch.where(
            plateau[has_ramp],
            torch.ones_like(ramp_values),
            ramp_values
        )

    if normalize_energy:
        # Energy / power normalization: each filter has unit squared norm
        norms = torch.sqrt(torch.sum(fb * fb, dim=1, keepdim=True).clamp_min(eps))
        fb = fb / norms

    return fb.to(dtype=dtype)


def _get_trapezoid_filterbank(
    n_fft: int,
    edges,
    overlap: float,
    slope_db: float,
    device: torch.device,
    dtype: torch.dtype,
    ) -> torch.Tensor:
    key = (
        n_fft,
        _hash_edges(edges),
        float(overlap),
        float(slope_db),
        str(device),
        str(dtype),
    )
    if key not in _trapezoid_basis_cache:
        _trapezoid_basis_cache[key] = _build_trapezoid_filterbank(
            n_fft=n_fft,
            edges=edges,
            overlap=overlap,
            slope_db=slope_db,
            device=device,
            dtype=dtype,
        )
    return _trapezoid_basis_cache[key]


# ============================================================
# STFT -> trapezoid-band spectrogram
# ============================================================

def trapezoid_spectrogram(
    y,
    n_fft,
    edges,
    sampling_rate,
    hop_size,
    win_size,
    overlap=0.25,
    center=False,
    normalize_fn=None,
    pad_mode="reflect",
    eps=1e-9,
):
    """
    Fast trapezoid-band spectrogram from waveform.

    Parameters
    ----------
    y : torch.Tensor
        Shape (B, T) or (T,).
    n_fft : int
        FFT size.
    edges : array-like, shape (n_filters + 1,)
        Local integer bin edges.
    sampling_rate : int
        Unused here directly, kept for API parity with mel_spectrogram.
    hop_size : int
        STFT hop length.
    win_size : int
        STFT window length.
    overlap : float
        Side ramp fraction of each band width.
    center : bool
        Passed to torch.stft.
    normalize_fn : callable or None
        Optional post-processing, e.g. spectral_normalize_torch.
    pad_mode : str
        Padding mode for the waveform.
    eps : float
        Magnitude floor.

    Returns
    -------
    spec_bands : torch.Tensor
        Shape (B, n_filters, n_frames) if input is batched,
        or (n_filters, n_frames) if input was 1D.
    """
    if y.dim() == 1:
        y = y.unsqueeze(0)
        squeeze_batch = True
    elif y.dim() == 2:
        squeeze_batch = False
    else:
        raise ValueError("y must have shape (T,) or (B, T)")

    device = y.device
    dtype =  torch.float32

    if y.dtype != dtype:
        y = y.to(dtype)

    filterbank = _get_trapezoid_filterbank(
        n_fft=n_fft,
        edges=edges,
        overlap=overlap,
        device=device,
        dtype=dtype,
    )
    window = _get_hann_window(win_size=win_size, device=device, dtype=dtype)

    # Match the padding style from the mel example when center=False
    if not center:
        pad = int((n_fft - hop_size) / 2)
        if pad > 0:
            y = F.pad(y.unsqueeze(1), (pad, pad), mode=pad_mode).squeeze(1)

    # return_complex=True is faster and cleaner
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=False,
        onesided=True,
        return_complex=True,
    )  # (B, K, frames)

    spec = spec.abs().clamp_min_(eps)

    # (F, K) @ (B, K, T) -> (B, F, T)
    spec = torch.matmul(filterbank, spec)

    if normalize_fn is not None:
        spec = normalize_fn(spec)

    if squeeze_batch:
        spec = spec.squeeze(0)

    return spec


# ============================================================
# Single-FFT mode
# ============================================================

def trapezoid_filter_fft(
    fft_input,
    edges,
    overlap=0.25,
    n_fft=None,
    normalize_fn=None,
    eps=1e-9,
):
    """
    Apply the cached trapezoid filterbank to a single FFT or a batch of FFTs.

    Parameters
    ----------
    fft_input : torch.Tensor
        One of:
        - shape (K,)                 : single FFT magnitude or complex rFFT
        - shape (..., K)            : batch of FFT magnitudes / complex rFFTs
    edges : array-like
        Local integer bin edges, length n_filters + 1.
    overlap : float
        Side ramp fraction.
    n_fft : int or None
        If None, inferred as 2 * (K - 1).
    normalize_fn : callable or None
        Optional post-processing.
    eps : float
        Magnitude floor.

    Returns
    -------
    bands : torch.Tensor
        Shape (..., n_filters)
    """
    if fft_input.dim() < 1:
        raise ValueError("fft_input must have at least 1 dimension")

    device = fft_input.device

    if torch.is_complex(fft_input):
        mag = fft_input.abs().clamp_min_(eps)
        dtype = mag.dtype
    else:
        mag = fft_input
        dtype = mag.dtype if mag.dtype in (torch.float32, torch.float64) else torch.float32
        if mag.dtype != dtype:
            mag = mag.to(dtype)

    k_bins = mag.shape[-1]
    if n_fft is None:
        n_fft = 2 * (k_bins - 1)

    expected_k = n_fft // 2 + 1
    if k_bins != expected_k:
        raise ValueError(
            f"fft_input last dimension ({k_bins}) does not match n_fft//2+1 ({expected_k})"
        )

    filterbank = _get_trapezoid_filterbank(
        n_fft=n_fft,
        edges=edges,
        overlap=overlap,
        device=device,
        dtype=dtype,
    )  # (F, K)

    # (..., K) @ (K, F) -> (..., F)
    bands = torch.matmul(mag, filterbank.transpose(0, 1))

    if normalize_fn is not None:
        bands = normalize_fn(bands)

    return bands