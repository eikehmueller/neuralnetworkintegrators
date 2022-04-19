"""Tests of neural network models"""
import pytest
import numpy as np
import tensorflow as tf
from dynamical_system import *  # pylint: disable=import-error,wildcard-import
from time_integrator import *  # pylint: disable=import-error,wildcard-import
from models import *  # pylint: disable=import-error,wildcard-import
from fixtures import *  # pylint: disable=import-error,wildcard-import


@pytest.mark.parametrize(
    "dynamical_system_name",
    [
        "harmonic_oscillator",
        "coupled_harmonic_oscillators",
        "coupled_pendulums",
    ],
)
def test_verlet_model(dynamical_system_name, request):
    """Check that integrating with the Verlet model gives the same results as the corresponding
    time integrator

    The equations of motion for the Verlet model are derived using autodiff, providing the
    integrator with explicit expressions for the kinetic and potential energy
    """
    tolerance = 1.0e-12
    dt = 0.1
    dynamical_system = request.getfixturevalue(dynamical_system_name)
    dim = dynamical_system.dim
    q0 = np.random.uniform(low=0, high=1, size=dim)
    p0 = np.zeros(dim)
    verlet_integrator = VerletIntegrator(dynamical_system, dt)
    nn_verlet_integrator = VerletModel(dim, dt, None, None)

    dynamical_system.backend = tf
    nn_verlet_integrator.V_pot = dynamical_system.V_pot
    nn_verlet_integrator.T_kin = dynamical_system.T_kin
    n_steps = 10
    verlet_integrator.set_state(q0, p0)
    verlet_integrator.integrate(n_steps)
    q_nn = np.array(q0)
    p_nn = np.array(p0)
    for _ in range(n_steps):
        q_nn, p_nn = nn_verlet_integrator.step(q_nn, p_nn)
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
def test_strang_splitting_model(dynamical_system_name, request):
    """Check that integrating with the Strang Splitting model gives the same results as the
    corresponding time integrator

    The equations of motion for the Verlet model are derived using autodiff, providing the
    integrator with explicit expressions for the kinetic and potential energy
    """
    np.random.seed(461857)
    tolerance = 1.0e-12
    dt = 0.1
    dynamical_system = request.getfixturevalue(dynamical_system_name)
    dim = dynamical_system.dim
    q0 = np.random.uniform(low=0, high=1, size=dim)
    p0 = np.zeros(dim)
    strang_splitting_integrator = StrangSplittingIntegrator(dynamical_system, dt)
    nn_strang_splitting_integrator = StrangSplittingModel(dim, dt, None)

    dynamical_system.backend = tf
    nn_strang_splitting_integrator.Hamiltonian = dynamical_system.energy
    n_steps = 1
    strang_splitting_integrator.set_state(q0, p0)
    strang_splitting_integrator.fast_code = False
    strang_splitting_integrator.integrate(n_steps)
    q_nn = np.array(q0)
    p_nn = np.array(p0)
    for _ in range(n_steps):
        q_nn, p_nn = nn_strang_splitting_integrator.step(q_nn, p_nn)
    diff = np.zeros(2 * dim)
    diff[:dim] = q_nn.numpy()[:] - strang_splitting_integrator.q[:]
    diff[dim:] = p_nn.numpy()[:] - strang_splitting_integrator.p[:]
    assert np.linalg.norm(diff) < tolerance
