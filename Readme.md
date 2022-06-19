[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# Neural-network based integrator for dynamical systems

This code implements both an s-step method for integrating dynamical systems of the form dy/dt = N(y(t)) and a single-step method with a Hamiltonian neural network model. Given the solution y(t) at s previous points in time, the value of the solution is predicted at the next timestep. For this, the correction is parametrised with a neural network, following the ideas in
[https://arxiv.org/abs/2004.06493](https://arxiv.org/abs/2004.06493) and [https://arxiv.org/abs/1906.01563](https://arxiv.org/abs/1906.01563). The code uses either a simple dense network or a LSTM architecture. A simple Verlet integrator is used for training.

## Mathematical details
Further mathematical details can be found in [this notebook](NNIntegrators.ipynb).

## Automatic C-code generation

## Code structure
### Notebooks
* [src/VisualiseIntegrators.ipynb](src/VisualiseIntegrators.ipynb)
* [src/TrainNNIntegrators.ipynb](src/TrainNNIntegrators.ipynb)
* [src/EvaluateNNIntegrators.ipynb](src/EvaluateNNIntegrators.ipynb)
* [src/VisualiseLossHistories.ipynb](src/VisualiseLossHistories.ipynb)
### Python modules
* [src/auxilliary.py](src/auxilliary.py)
* [src/models.py](src/models.py)
* [src/dynamical_system.py](src/dynamical_system.py)
* [src/data_generator.py](src/data_generator.py)
* [src/time_integrator.py](src/time_integrator.py)
* [src/nn_integrator.py](src/nn_integrator.py)

## Testing
A set of unit tests which can be run with `pytest` are collected in the directory `src/tests`.


