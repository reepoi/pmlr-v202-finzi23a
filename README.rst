.. note::

   *This is a fork of the official code.*
   Find the official code `here <https://github.com/google-research/google-research/tree/9dda2b5e6503284eeb24e746d4103ed37019a80e/simulation_research/diffusion>`_.

User-defined event sampling and uncertainty quantification in diffusion models for physical dynamical systems
=============================================================================================================

*This is not an official Google product*

This repository contains exploratory code for conditioning techniques for
sampling from diffusion models. This is particularly geared towards events that
arise in the prediction of future trajectories of dynamical systems. We study
several dynamical systems including non-linear pendulums and the Lorenz system.

This is a work in progress; and the results would be presented in an upcoming
research paper.


Installation
============

#. Install ``uv``:

   .. code:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh

#. **Suggested:** Set the package cache directory of ``uv`` to a directory in a mounted drive.
   For example,

   .. code:: bash

      echo "export UV_CACHE_DIR=/root/workspace/out/uv-cache" >> ~/.bashrc
      source ~/.bashrc

#. Install ``cuDNN``:

   .. code:: bash

      apt-get install -y cudnn9-cuda-12

#. Install Python dependencies using ``uv``:

   .. code:: bash

      uv sync

Training the models
===================

To see all the configurable options, run

.. code:: bash

   python src/userfm/main.py dataset=GaussianMixture model=ModelDiffusion -c job

Change the default ``out_dir`` of ``class Config`` in ``src/cs.py`` to where you want the trained models to be saved, or override it with

.. code:: bash

   python src/userfm/main.py out_dir=<path/to/save/trained/models> <other_overrides...>

Gaussian Mixture
^^^^^^^^^^^^^^^^

To train the VE-SDE diffusion model, run

.. code:: bash

   python src/userfm/main.py +experiment=GaussianMixture model=ModelDiffusion

To train the flow matching model, run

.. code:: bash

   python src/userfm/main.py +experiment=GaussianMixture model=ModelFlowMatching model/conditional_flow=<conditional_flow>

where ``<conditional_flow>`` is replaced with either ``ConditionalOT`` or ``MinibatchOTConditionalOT``.

Simple Harmonic Oscillator
^^^^^^^^^^^^^^^^^^^^^^^^^^

To train the VE-SDE diffusion model, run

.. code:: bash

   python src/userfm/main.py +experiment=SimpleHarmonicOscillator model=ModelDiffusion

To train the flow matching model, run

.. code:: bash

   python src/userfm/main.py +experiment=SimpleHarmonicOscillator model=ModelFlowMatching model/conditional_flow=<conditional_flow>

where ``<conditional_flow>`` is replaced with either ``ConditionalOT`` or ``MinibatchOTConditionalOT``.

Lorenz
^^^^^^

To train the VE-SDE diffusion model, run

.. code:: bash

   python src/userfm/main.py +experiment=Lorenz model=ModelDiffusion

To train the flow matching model, run

.. code:: bash

   python src/userfm/main.py +experiment=Lorenz model=ModelFlowMatching model/conditional_flow=<conditional_flow>

where ``<conditional_flow>`` is replaced with either ``ConditionalOT`` or ``MinibatchOTConditionalOT``.

Fitzhugh-Nagumo
^^^^^^^^^^^^^^^

To train the VE-SDE diffusion model, run

.. code:: bash

   python src/userfm/main.py +experiment=FitzHughNagumo model=ModelDiffusion

To train the flow matching model, run

.. code:: bash

   python src/userfm/main.py +experiment=FitzHughNagumo model=ModelFlowMatching model/conditional_flow=<conditional_flow>

where ``<conditional_flow>`` is replaced with either ``ConditionalOT`` or ``MinibatchOTConditionalOT``.

Original Code (mostly)
----------------------

Lorenz
^^^^^^
Change ``--workdir`` as needed.

.. code:: bash

   python src/userdiffusion/main.py --config=src/userdiffusion/config.py --config.dataset=LorenzDataset --workdir=../../out/diffusion-dynamics/pmlr-v202-finzi23a/runs/lorenz/

Fitzhugh-Nagumo
^^^^^^^^^^^^^^^
Change ``--workdir`` as needed.

.. code:: bash

   python src/userdiffusion/main.py --config=src/userdiffusion/config.py --config.dataset=FitzHughDataset --workdir=../../out/diffusion-dynamics/pmlr-v202-finzi23a/runs/fitzhugh/

Pendulum
^^^^^^^^
Change ``--workdir`` as needed.

.. code:: bash

   python src/userdiffusion/main.py --config=src/userdiffusion/config.py --config.dataset=NPendulum --workdir=../../out/diffusion-dynamics/pmlr-v202-finzi23a/runs/pendulum/


Evaluating the models
=====================

Run the respective Jupyter notebook ``notebooks/plots_[lorenz,fitzhugh,pendulum]`` to produce some of the plots in the paper.
Change the ``workdir`` variable as needed.
Note that not all the code for the plots works.
