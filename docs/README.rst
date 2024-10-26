References
----------

-  `Kurashige, Yuki. “Matrix product state formulation of the
   multiconfiguration time-dependent Hartree theory.” The Journal of
   Chemical Physics 149.19 (2018):
   194114. <https://aip.scitation.org/doi/abs/10.1063/1.5051498>`__

   -  Time evolution algorithm of MPS
   -  Population dynamics

-  `Hino, Kentaro, and Yuki Kurashige. “Matrix Product State Formulation
   of the MCTDH Theory in Local Mode Representations for Anharmonic
   Potentials.” Journal of Chemical Theory and Computation 18.6 (2022):
   3347-3356. <https://pubs.acs.org/doi/abs/10.1021/acs.jctc.2c00243>`__

   -  Spectroscopy
   -  Anharmonic potential
   -  Local mode vibration

-  `Hino, Kentaro, and Yuki Kurashige. “Encoding a Many-body Potential
   Energy Surface into a Grid-Based Matrix Product Operator.” Journal of
   Chemical Theory and Computation 20.9 (2024):
   3839-3849.) <https://pubs.acs.org/doi/10.1021/acs.jctc.4c00046>`__

   -  Grid-based MPO
   -  DVR-MPS
   -  nMR

-  `Beck, Michael H., et al. “The multiconfiguration time-dependent
   Hartree (MCTDH) method: a highly efficient algorithm for propagating
   wavepackets.” Physics reports 324.1 (2000):
   1-105. <https://www.sciencedirect.com/science/article/pii/S0370157399000472>`__

   -  MCTDH paper
   -  DVR is also written.

Installation
------------

-  The easiest way to install ``pytdscf`` is to use ``pip``.

   Prepare Python 3.10 or later and execute;

   .. code:: bash

      $ python -m venv pytdscf-env
      $ source pytdscf-env/bin/activate
      $ pip install git+https://github.com/QCLovers/PyTDSCF

-  We recommend install ``pytdscf`` from source using
   ```uv`` <https://docs.astral.sh/uv/>`__

   .. code:: bash

      $ git clone https://github.com/QCLovers/PyTDSCF.git
      $ cd PyTDSCF
      $ uv version
      uv 0.4.18 (7b55e9790 2024-10-01)
      $ uv sync --all-extras

   will install all dependencies including development tools. If you
   need only the runtime dependencies, you can use ``uv sync --no-dev``.

   Then, you can execute ``pytdscf`` by

   .. code:: bash

      $ uv run python xxx.py

   or

   .. code:: bash

      $ souce .venv/bin/activate
      $ python
      >>> import pytdscf

   For jupyter notebook tutorials, you can use

   .. code:: bash

      $ uv run jupyter lab

For GPU users
~~~~~~~~~~~~~

``pytdscf`` works both on CPU and GPU. If you treat large-scale batch or
model, we recommend using GPU. See also `JAX’s GPU
support <https://jax.readthedocs.io/en/latest/installation.html>`__.

1. Make sure the latest NVIDIA driver is installed.

   .. code:: bash

      $ /usr/local/cuda/bin/nvcc -V
      nvcc: NVIDIA (R) Cuda compiler driver
      Copyright (c) 2005-2024 NVIDIA Corporation
      Built on Wed_Apr_17_19:19:55_PDT_2024
      Cuda compilation tools, release 12.5, V12.5.40
      Build cuda_12.5.r12.5/compiler.34177558_0

2. Install GPU-supported JAX in your virtual envirionment.

   .. code:: bash

      $ uv pip install -U "jax[cuda12]"
      $ uv run python -c "import jax; print(jax.default_backend())"
      'gpu'

Testing
~~~~~~~

.. code:: bash

   $ cd tests/build
   $ uv run pytest ..

For developers
~~~~~~~~~~~~~~

You should install pre-commit hooks including ruff formatting and
linting, mypy type checking, pytest testing, and so on.

.. code:: bash

   $ uv run pre-commit install
   $ git add .
   $ uv run pre-commit

Before push, you must fix problems!!

Please feel free to give us feedback or pull requests.

How to run
----------

See quick-start example in
`documentation <https://qclovers.github.io/PyTDSCF/notebook/quick-start.html>`__
or ``test`` directory.
