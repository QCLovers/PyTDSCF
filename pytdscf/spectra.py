"""Plot spectra from FFT of auto-correlation dat file"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate

from pytdscf import units


def load_autocorr(dat_file: str) -> tuple[np.ndarray, np.ndarray]:
    """Load auto-correlation data from dat file

    Args:
        dat_file (str): path to dat file
    Returns:
        Tuple[np.ndarray, np.ndarray]: (time in fs, auto-correlation data)

    """
    with open(dat_file, "r") as f:
        # Read first line to alarm time unit
        first_line = f.readline()
        if "fs" not in first_line:
            print("WARNING: time unit is not fs")
        data = np.loadtxt(f, usecols=(0, 1), skiprows=0, dtype=np.complex128)
        time_fs = data[:, 0].real
        autocorr = data[:, 1]
    if time_fs[0] != 0.0:
        raise ValueError(f"time is not starting from 0.0 but {time_fs[0]}")
    if autocorr[0] != 1.0:
        raise ValueError(
            f"auto-correlation at t=0 is not 1.0 but {autocorr[0]}"
        )
    return (time_fs, autocorr)


def _multiply_window(
    time_fs: np.ndarray, autocorr: np.ndarray, window: str = "cos2"
) -> np.ndarray:
    """Multiply window function to auto-correlation data

    Args:
        time_fs (np.ndarray): time in fs
        autocorr (np.ndarray): auto-correlation data
        window (str, optional): window function. Defaults to "cos2".

    Returns:
        np.ndarray: auto-correlation data multiplied by window function

    """
    if window == "cos2":
        """cos^2(tπ/2T) (0 < t < T)"""
        window = np.cos(np.pi * time_fs / time_fs[-1] / 2) ** 2
    elif window == "cos":
        window = np.cos(np.pi * time_fs / time_fs[-1] / 2)
    elif window is None:
        window = np.ones_like(time_fs)
    else:
        raise ValueError(f"window function {window} is not defined")
    return autocorr * window


def ifft_autocorr(
    time_fs: np.ndarray,
    autocorr: np.ndarray,
    E_shift: float = 0.0,
    window: str = "cos2",
    power: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse FFT of auto-correlation data

    Args:
        time_fs (np.ndarray): time in fs
        autocorr (np.ndarray): auto-correlation data
        E_shift (float, optional): energy shift in eV. we often use ZPE. Defaults to 0.0.
        window (str, optional): window function. Defaults to "cos2".
        power (bool, optional): if True, intensity is power spectrum.\
            Defaults to False, thus IR absorption spectrum in arb. unit. and energy is shifted.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (wave number in cm-1, intensity)

    """
    func = interpolate.interp1d(time_fs, autocorr, kind="cubic")
    Δt = np.amax(time_fs[1:-1] - time_fs[0:-2]) / 2
    print(
        f"delta_t: {Δt:6.3f}[fs], max_freq: {2 * np.pi * 5308.837451 * 1.0 / Δt:6.1f}[cm-1]"
    )
    N = int((time_fs[-1] - time_fs[0]) / Δt)
    time_unif = np.arange(N) * Δt
    data_unif = func(time_unif)
    autocorr_with_window = _multiply_window(time_unif, data_unif, window)
    ω_cm1 = -np.fft.fftshift(np.fft.fftfreq(N, Δt)) * 1e15 * (3.33564 * 1e-11)
    intensity = np.fft.fftshift(np.fft.fft(autocorr_with_window) * Δt)
    ω_cm1 = np.flipud(ω_cm1)
    if power:
        intensity = np.flipud(intensity.real)
    else:
        ω_cm1 -= E_shift * units.au_in_cm1 / units.au_in_eV
        intensity = np.flipud(intensity.real) * ω_cm1
    return (ω_cm1, intensity)


def export_spectrum(wave_number, intensity, filename: str = "spectrum.dat"):
    """Export spectrum to dat file

    Args:
        wave_number (np.ndarray): wave number in cm-1
        intensity (np.ndarray): intensity
        filename (str, optional): path to dat file. Defaults to "spectrum.dat".

    """
    with open(filename, "w") as f:
        f.write("# wave_number[cm-1]\t intensity[arb. unit]\n")
        data = np.hstack(
            (wave_number.reshape((-1, 1)), intensity.reshape((-1, 1)))
        )
        np.savetxt(f, data, fmt="%15.8f", delimiter="\t")


def plot_autocorr(time_fs: np.ndarray, autocorr: np.ndarray, gui: bool = True):
    """Plot auto-correlation data

    Args:
        time_fs (np.ndarray): time in fs
        autocorr (np.ndarray): auto-correlation data
        gui (bool, optional): if True, plt.show() use GUI. Defaults to True.

    """
    plt.figure(figsize=(15, 6), dpi=80)
    plt.rcParams["font.size"] = 16
    plt.xlabel("Re & Im of autocorr")
    plt.plot(time_fs, autocorr.real, color="blue", label="real")
    plt.plot(time_fs, autocorr.imag, color="red", label="imag")
    plt.xlabel("time[fs]")
    plt.title("auto-corr <Ψ(0)|Ψ(t)>")
    plt.legend()
    plt.show(block=gui)


def plot_spectrum(
    wave_number: np.ndarray,
    intensity: np.ndarray,
    lower_bound: float = 0.0,
    upper_bound: float = 4000.0,
    filename: str = "spectrum.pdf",
    export: bool = True,
    show_in_eV: bool = False,
    gui: bool = True,
    normalize: bool = True,
):
    """Plot spectrum

    Args:
        wave_number (np.ndarray): wave number in cm-1
        intensity (np.ndarray): intensity
        lower_bound (float, optional): lower bound of wave number in cm-1. \
            Defaults to 0.0.
        upper_bound (float, optional): upper bound of wave number in cm-1. \
            Defaults to 4000.0.
        filename (str): path to figure. Defaults to "spectrum.pdf".
        export (bool): export spectrum to dat file. Defaults to True.
        show_in_eV (bool): show in eV. Defaults to False.
        gui (bool): if True, plt.show() use GUI. Defaults to True.
        normalize (bool): normalize intensity in range of between lower and upper bound. Defaults to True.

    """
    lower_bound_index = np.searchsorted(wave_number, lower_bound)
    upper_bound_index = np.searchsorted(wave_number, upper_bound)

    if normalize:
        intensity = (intensity / max(intensity))[
            lower_bound_index - 1 : upper_bound_index + 1
        ]
    else:
        intensity = intensity[lower_bound_index - 1 : upper_bound_index + 1]
    wave_number = wave_number[lower_bound_index - 1 : upper_bound_index + 1]

    func = interpolate.interp1d(wave_number, intensity, kind="cubic")

    x = np.arange(lower_bound, upper_bound, 1, dtype=np.float64)
    y = func(x)
    if export:
        parts = filename.split(".")
        if len(parts) == 1:
            dat_filename = filename + ".dat"
        else:
            parts[-1] = "dat"
            dat_filename = ".".join(parts)
        export_spectrum(x, y, filename=dat_filename)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.rcParams["font.size"] = 16
    plt.title("IR absorption spectrum")
    if show_in_eV:
        x *= units.au_in_eV / units.au_in_cm1
        plt.xlabel("wave number / eV")
        plt.xlim(
            lower_bound * units.au_in_eV / units.au_in_cm1,
            upper_bound * units.au_in_eV / units.au_in_cm1,
        )
    else:
        plt.xlabel("wave number / cm$^{-1}$")
        plt.xlim(lower_bound, upper_bound)
    plt.plot(x, y, "-", color="red", linewidth=3)
    plt.ylabel("intensity / arb. unit")
    if normalize:
        plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show(block=gui)


if __name__ == "__main__":
    time, autocorr = load_autocorr(sys.argv[1])
    plot_autocorr(time, autocorr)
    freq, intensity = ifft_autocorr(time, autocorr)
    plot_spectrum(
        freq, intensity, lower_bound=-1000, upper_bound=10000, show_in_eV=True
    )
