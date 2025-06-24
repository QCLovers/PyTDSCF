[![Deploy static content to Pages](https://github.com/QCLovers/PyTDSCF/actions/workflows/static.yml/badge.svg)](https://github.com/QCLovers/PyTDSCF/actions/workflows/static.yml)
[![unittest](https://github.com/QCLovers/PyTDSCF/actions/workflows/unittest.yml/badge.svg)](https://github.com/QCLovers/PyTDSCF/actions/workflows/unittest.yml)
# PyTDSCF
![reduced_density_bath](https://github.com/user-attachments/assets/a0bf7f6c-0b43-48a5-8e2b-36bd5436fbde)

PyTDSCF is a Python package for high-dimensional wave-packet dynamics simulations based on tensor train.

## Features

### Multiple MPO Types for Hamiltonian Representation

You can use various types of Matrix Product Operators (MPO) as Hamiltonians:


- **[Symbolic MPO](https://qclovers.github.io/PyTDSCF/notebook/poly-MPO-H2O-relax.html)**
  ![image](https://github.com/user-attachments/assets/b3c8e646-b5a2-4ff9-8c56-46cde369c76d)

- **[Grid-based MPO](https://qclovers.github.io/PyTDSCF/notebook/grid-based-MPO-H2CO.html)**
  ![image](https://github.com/user-attachments/assets/23e5123e-ce6b-4a15-8de8-a47618a07ae3)

- **[Neural network MPO](https://github.com/KenHino/Pompon)**
  ![image](https://github.com/user-attachments/assets/5a9de281-3829-4a5e-8913-936328d7f734)

### Flexible Basis Sets

Support for various basis types:
- Boson states $|n\rangle$
- DVR grid states $|x\rangle$
- Spin states $|s\rangle$
- Exciton states $|e\rangle$
- ... (whatever)

### Simulation Capabilities

PyTDSCF enables large-dimensional system simulations for:
- Vibrational ground state calculations
- Autocorrelation functions
- IR spectroscopy
- Nonadiabatic population dynamics
- Time-dependent expectation values
- Non-Markovian open quantum dynamics
- Reduced density matrix analysis
- Liouville space dissipation
- And more...

### Performance Features

- **GPU acceleration** through JAX for large-scale calculations
- **Parallel execution** for ab initio potential energy surface calculations

## Documentation

Comprehensive documentation is available [here](https://qclovers.github.io/PyTDSCF/notebook/quick-start.html)!

## References

- [Kurashige, Yuki. "Matrix product state formulation of the multiconfiguration time-dependent Hartree theory." The Journal of Chemical Physics 149.19 (2018): 194114.](https://aip.scitation.org/doi/abs/10.1063/1.5051498)
  - Time evolution algorithm of MPS
  - Population dynamics
- [Hino, Kentaro, and Yuki Kurashige. "Matrix Product State Formulation of the MCTDH Theory in Local Mode Representations for Anharmonic Potentials." Journal of Chemical Theory and Computation 18.6 (2022): 3347-3356.](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.2c00243)
  - Spectroscopy
  - Anharmonic potential
  - Local mode vibration
- [Hino, Kentaro, and Yuki Kurashige. "Encoding a Many-body Potential Energy Surface into a Grid-Based Matrix Product Operator." Journal of Chemical Theory and Computation 20.9 (2024): 3839-3849.)](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00046)
  - Grid-based MPO
  - DVR-MPS
  - nMR
- [Hino, Kentaro, and Yuki Kurashige. "Neural network matrix product operator: A multi-dimensionally integrable machine learning potential." Physical Review Research 7.2 (2025): 023217.](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.7.023217)
  - Neural network MPO

## Installation

### Recommended: Install from Source using `uv`

We recommend installing `pytdscf` from source using [`uv`](https://docs.astral.sh/uv/):

```bash
$ git clone https://github.com/QCLovers/PyTDSCF.git
$ cd PyTDSCF
$ uv version
uv 0.5.4 (c62c83c37 2024-11-20)
$ uv sync --all-extras
```

This will install all dependencies including development tools.
If you only need runtime dependencies, use `uv sync --no-dev`.

You can then run `pytdscf` using:

```bash
$ uv run python xxx.py
```

Or activate the virtual environment:

```bash
$ source .venv/bin/activate
$ python
>>> import pytdscf
```

For Jupyter notebook tutorials:

```bash
$ uv run jupyter lab
```

### Alternative: Install via pip

The easiest way to install `pytdscf` is using `pip`:

Prepare Python 3.10 or later and execute:

```bash
$ python -m venv pytdscf-env
$ source pytdscf-env/bin/activate
$ pip install git+https://github.com/QCLovers/PyTDSCF
```

### GPU Support

`pytdscf` works on both CPU and GPU.
For large-scale batch processing or complex models, we recommend using GPU.
See also [JAX's GPU support](https://jax.readthedocs.io/en/latest/installation.html).

1. Ensure the latest NVIDIA driver is installed:

    ```bash
    $ /usr/local/cuda/bin/nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2024 NVIDIA Corporation
    Built on Wed_Apr_17_19:19:55_PDT_2024
    Cuda compilation tools, release 12.5, V12.5.40
    Build cuda_12.5.r12.5/compiler.34177558_0
    ```

2. Install GPU-supported JAX in your virtual environment:

    ```bash
    $ uv pip install -U "jax[cuda12]"
    $ uv run python -c "import jax; print(jax.default_backend())"
    'gpu'
    ```

## Testing

```bash
$ cd tests/build
$ uv run pytest ..
```

## Development

We welcome feedback and pull requests. For developers, install pre-commit hooks including ruff formatting and linting, mypy type checking, pytest testing, and more:

```bash
$ uv run pre-commit install
$ git add .
$ uv run pre-commit
```

**Important**: Fix any issues before pushing!

We welcome feedback and pull requests.

## Getting Started

See the quick-start example in our [documentation](https://qclovers.github.io/PyTDSCF/notebook/quick-start.html) or explore the `test` directory for examples.
