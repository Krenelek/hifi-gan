from adaptive_filter.constants import ISO226_F, ISO226_AF, ISO226_LU, ISO226_TF
import numpy as np


def iso226(phon):
    Ln = phon
    Af = 4.47e-3 * (10**(0.025 * Ln) - 1.15) + \
         (0.4 * 10**(((ISO226_TF + ISO226_LU) / 10) - 9))**ISO226_AF
    spl = (10 / ISO226_AF) * np.log10(Af) - ISO226_LU + 94
    return ISO226_F, spl



def relative_sensitivity_gain_curve(fs=22050, phon=60, n_bins=512):
    """
    Build a relative sensitivity gain curve from ISO 226.
    The curve is normalized to 0 dB at 1 kHz:
    positive values: more sensitive than 1 kHz
    negative values: less sensitive than 1 kHz
    """

    freqs_fft = np.linspace(0, fs / 2, n_bins)

    f_iso, spl_iso = iso226(phon)

    curve_db = np.interp(
        freqs_fft,
        f_iso,
        spl_iso,
        left=spl_iso[0],
        right=spl_iso[-1],
    )

    # norm: 1 kHz = 0 dB
    ref_spl = np.interp(1000.0, f_iso, spl_iso)
    curve_db = -(curve_db - ref_spl)

    return curve_db


def apply_relative_sensitivity_to_fft_dict(
    fft_dict,
    phonemes=None,
    fs=22050,
    phon=60,
    eps=1e-12,
    log_data = False,
):
    """
    fft_dict:
        key: str (phoneme)
        value: np.ndarray shape (bit_end-bit-start,) - amplitude lin/log FFT

    out:
        key: str (phoneme)
        value: np.ndarray shape (bit_end-bit-start,) - amplitude fon^-1 FFT
    """
    curve_db = None

    out = {}

    for phoneme, fft in fft_dict.items():
        if phonemes is not None and phoneme not in phonemes:
            continue
        if curve_db is None:
            curve_db = relative_sensitivity_gain_curve(fs=fs, phon=phon, n_bins=len(fft))
        if not log_data:
            fft = 20 * np.log10(fft + eps)

        fft_db_weighted = fft + curve_db

        out[phoneme] = fft_db_weighted

    return out
