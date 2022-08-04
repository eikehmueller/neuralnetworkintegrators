"""Tests of neural network models"""
import inspect
import pytest
import numpy as np
import tensorflow as tf
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures import *  # pylint: disable=import-error,wildcard-import
from dynamical_system import *  # pylint: disable=import-error,wildcard-import
from time_integrator import *  # pylint: disable=import-error,wildcard-import
from models import *  # pylint: disable=import-error,wildcard-import


@pytest.mark.parametrize(
    "dynamical_system_name",
    [
        "harmonic_oscillator",
        "coupled_harmonic_oscillators",
        "coupled_pendulums",
    ],
)
def test_verlet_model(dynamical_system_name, request, monkeypatch):
    """Check that integrating with the Verlet model gives the same results as the corresponding
    time integrator

    The equations of motion for the Verlet model are derived using autodiff, providing the
    integrator with explicit expressions for the kinetic and potential energy
    """
    np.random.seed(461857)
    tolerance = 1.0e-6
    dt = 0.1
    dynamical_system = request.getfixturevalue(dynamical_system_name)
    # replace numpy by tensorflow
    monkeypatch.setattr(inspect.getmodule(dynamical_system), "np", tf)
    dim = dynamical_system.dim
    q0 = np.random.uniform(low=0, high=1, size=dim)
    p0 = np.random.uniform(low=0, high=1, size=dim)
    verlet_integrator = VerletIntegrator(dynamical_system, dt)
    nn_verlet_model = VerletModel(dim, dt, None, None)
    nn_verlet_model.V_pot = dynamical_system.V_pot
    nn_verlet_model.T_kin = dynamical_system.T_kin
    n_steps = 4
    verlet_integrator.set_state(q0, p0)
    verlet_integrator.integrate(n_steps)
    q_nn = np.array(q0, dtype=np.float32)
    p_nn = np.array(p0, dtype=np.float32)
    for _ in range(n_steps):
        q_nn, p_nn = nn_verlet_model.step(q_nn, p_nn)
    diff = np.zeros(2 * dim)
    diff[:dim] = q_nn.numpy()[:] - verlet_integrator.q[:]
    diff[dim:] = p_nn.numpy()[:] - verlet_integrator.p[:]
    assert np.linalg.norm(diff) < tolerance


@pytest.mark.parametrize(
    "dynamical_system_name",
    [
        "harmonic_oscillator",
        "coupled_harmonic_oscillators",
        "double_pendulum",
        "coupled_pendulums",
    ],
)
def test_strang_splitting_model(dynamical_system_name, request, monkeypatch):
    """Check that integrating with the Strang Splitting model gives the same results as the
    corresponding time integrator

    The equations of motion for the Verlet model are derived using autodiff, providing the
    integrator with an explicit expressions for the Hamiltonian
    """
    np.random.seed(461857)
    tolerance = 1.0e-6
    dt = 0.1
    dynamical_system = request.getfixturevalue(dynamical_system_name)
    # replace numpy by tensorflow
    monkeypatch.setattr(inspect.getmodule(dynamical_system), "np", tf)
    dim = dynamical_system.dim
    q0 = np.random.uniform(low=0, high=1, size=dim)
    p0 = np.random.uniform(low=0, high=1, size=dim)
    x0 = np.random.uniform(low=0, high=1, size=dim)
    y0 = np.random.uniform(low=0, high=1, size=dim)
    strang_splitting_integrator = StrangSplittingIntegrator(dynamical_system, dt)
    nn_strang_splitting_model = StrangSplittingModel(4 * dim, dt, None)
    nn_strang_splitting_model.Hamiltonian = dynamical_system.energy
    n_steps = 4
    strang_splitting_integrator.set_state(q0, p0)
    strang_splitting_integrator.set_extended_state(x0, y0)
    strang_splitting_integrator.integrate(n_steps)
    q_nn = np.array(q0, dtype=np.float32)
    p_nn = np.array(p0, dtype=np.float32)
    x_nn = np.array(x0, dtype=np.float32)
    y_nn = np.array(y0, dtype=np.float32)
    for _ in range(n_steps):
        q_nn, p_nn, x_nn, y_nn = nn_strang_splitting_model.step(q_nn, p_nn, x_nn, y_nn)
    diff = np.zeros(4 * dim)
    diff[:dim] = q_nn.numpy()[:] - strang_splitting_integrator.q[:]
    diff[dim : 2 * dim] = p_nn.numpy()[:] - strang_splitting_integrator.p[:]
    diff[2 * dim : 3 * dim] = x_nn.numpy()[:] - strang_splitting_integrator.x[:]
    diff[3 * dim :] = y_nn.numpy()[:] - strang_splitting_integrator.y[:]
    assert np.linalg.norm(diff) < tolerance
