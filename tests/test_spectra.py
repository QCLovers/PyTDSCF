import os

import pytest

import pytdscf


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_spectra():
    time, autocorr = pytdscf.spectra.load_autocorr(
        f"{os.path.dirname(os.path.abspath(__file__))}/autocorr.dat"
    )
    pytdscf.spectra.plot_autocorr(time, autocorr, gui=False)
    freq, intensity = pytdscf.spectra.ifft_autocorr(time, autocorr)
    pytdscf.spectra.plot_spectrum(
        freq,
        intensity,
        lower_bound=1000,
        upper_bound=4000,
        show_in_eV=True,
        gui=False,
    )


if __name__ == "__main__":
    test_spectra()
