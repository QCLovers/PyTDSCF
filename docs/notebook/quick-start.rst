=======================
Quick Start Examples
=======================


.. _Example 1: poly-MPO-H2O-relax.html
.. _Example 2: poly-MPO-H2O-operate.html
.. _Example 3: poly-MPO-H2O-propagate.html
.. _Example 4: spectra-H2O.html
.. _Example 5: LVC.html
.. _Example 6: henon_heiles_HO-DVR.html
.. _Example 7: henon_heiles_Sine-DVR.html
.. _Example 8: electronic-structure-calculation.html
.. _Example 9: grid-based-MPO-H2CO.html
.. _Example 10: TD_reduced_density.html
.. _Example 11: TD_reduced_density_exciton.html
.. _Example 12: singlet_fission.html
.. _Example 13: donor-acceptor.html
.. _Example 14: radicalpair-sse.html
.. _Example 15: radicalpair-liouville.html


+---------------+-------------+--------------+------------+------------------+------------+--------------------------------+
|               | Run Type    | Wavefunction | Potential  | Basis            | Backend    | misc.                          |
+===============+=============+==============+============+==================+============+================================+
| `Example 1`_  | Relaxation  | MPS          | Polynomial | HO-DVR           | Numpy      | H2O molecule                   |
+---------------+-------------+--------------+------------+------------------+------------+--------------------------------+
| `Example 2`_  | Operation   | MPS          | Polynomial | HO-DVR           | Numpy      | Restart from `Example 1`_      |
+---------------+-------------+--------------+------------+------------------+------------+--------------------------------+
| `Example 3`_  | Propagation | MPS          | Polynomial | HO-DVR           | Numpy      | Restart from `Example 2`_      |
+---------------+-------------+--------------+------------+------------------+------------+--------------------------------+
| `Example 5`_  | Propagation | MPS          | Polynomial | HO-FBR           | Numpy      | Linear vibronic coupling model |
+---------------+-------------+--------------+------------+------------------+------------+--------------------------------+
| `Example 6`_  | Propagation | MPS          | HDMR func  | HO-DVR           | Numpy      | Henon-Heiles Hamiltonian       |
+---------------+-------------+--------------+------------+------------------+------------+--------------------------------+
| `Example 7`_  | Propagation | MPS          | HDMR func  | Sine-DVR         | Numpy      | Henon-Heiles Hamiltonian       |
+---------------+-------------+--------------+------------+------------------+------------+--------------------------------+
| `Example 9`_  | Relaxation  | MPS          | HDMR grid  | HO-DVR           | JAX        | Restart from `Example 8`_      |
+---------------+-------------+--------------+------------+------------------+------------+--------------------------------+
| `Example 10`_ | Propagation | MPS          | HDMR func  | Sine,Exp-DVR     | JAX        | periodic boundary condition    |
+---------------+-------------+--------------+------------+------------------+------------+--------------------------------+
| `Example 11`_ | Propagation | MPS          | Symbolic   | Exciton+Sine+Exp | JAX        | Same model as `Example 10`_    |
+---------------+-------------+--------------+------------+------------------+------------+--------------------------------+
| `Example 12`_ | Propagation | MPS          | Symbolic   | Exciton, Boson   | Numpy      | Singlet fission + 183-D bath   |
+---------------+-------------+--------------+------------+------------------+------------+--------------------------------+
| `Example 13`_ | Propagation | MPS          | Symbolic   | Exciton, Boson   | Numpy      | Donor-Acceptor + 99-D bath     |
+---------------+-------------+--------------+------------+------------------+------------+--------------------------------+
| `Example 14`_ | Propagation | Stochastic   | Symbolic   | Spin             | Numpy      | Electron and nuclear spins     |
+---------------+-------------+--------------+------------+------------------+------------+--------------------------------+
| `Example 15`_ | Propagation | Liouville    | Symbolic   | Spin             | Numpy      | Electron and nuclear spins     |
+---------------+-------------+--------------+------------+------------------+------------+--------------------------------+


+---------------+----------------------------------+---------------------------+
|               | Run Type                         | misc.                     |
+===============+==================================+===========================+
| `Example 4`_  | Spectrum                         | Restart from `Example 3`_ |
+---------------+----------------------------------+---------------------------+
| `Example 8`_  | Electronic structure calculation | ASE is required           |
+---------------+----------------------------------+---------------------------+


.. toctree::
   :maxdepth: 1

   poly-MPO-H2O-relax.ipynb
   poly-MPO-H2O-operate.ipynb
   poly-MPO-H2O-propagate.ipynb
   spectra-H2O.ipynb
   LVC.ipynb
   henon_heiles_HO-DVR.ipynb
   henon_heiles_Sine-DVR.ipynb
   electronic-structure-calculation.ipynb
   grid-based-MPO-H2CO.ipynb
   TD_reduced_density.ipynb
   TD_reduced_density_exciton.ipynb
   singlet_fission.ipynb
   donor-acceptor.ipynb
   radicalpair-sse.ipynb

Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
