import os

import numpy as np
import pytest

import pytdscf

time, autocorr = pytdscf.spectra.load_autocorr(
    f"{os.path.dirname(os.path.abspath(__file__))}/autocorr.dat"
)
pytdscf.spectra.plot_autocorr(time, autocorr, gui=False)
freq, intensity = pytdscf.spectra.ifft_autocorr(time, autocorr)
assert pytest.approx(max(intensity)) == 28860.651565826236
assert pytest.approx(freq[np.argmax(intensity)]) == 2684.0796620397296


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_spectra_in_eV():
    pytdscf.spectra.plot_spectrum(
        freq,
        intensity,
        lower_bound=1000,
        upper_bound=4000,
        show_in_eV=True,
        gui=False,
    )

def test_spectra_in_nm():
    pytdscf.spectra.plot_spectrum(
        freq,
        intensity,
        lower_bound=1000,
        upper_bound=4000,
        show_in_nm=True,
        gui=False,
    )

if __name__ == "__main__":
    test_spectra_in_eV()
    test_spectra_in_nm()
