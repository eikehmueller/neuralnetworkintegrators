"""Fixtures used by all tests"""
import pytest
from dynamical_system import *  # pylint: disable=import-error,wildcard-import


@pytest.fixture()
def harmonic_oscillator():
    """Construct one dimensional harmonic oscillator instance"""
    mass = 1.2
    k_spring = 0.9
    return HarmonicOscillator(mass, k_spring)


@pytest.fixture()
def coupled_harmonic_oscillators():
    """Construct coupled harmonic oscillator instance"""
    mass = [1.2, 0.8]
    k_spring = [0.9, 1.1]
    k_spring_c = 0.4
    return CoupledHarmonicOscillators(mass, k_spring, k_spring_c)


@pytest.fixture()
def double_pendulum():
    """Construct double pendulum instance"""
    mass = [1.2, 0.8]
    L0, L1 = 0.8, 1.2
    return DoublePendulum(mass, L0, L1)


@pytest.fixture()
def coupled_pendulums():
    """Construct instance of coupled pendulums"""
    mass = 1.2
    L_rod = 0.9
    d_anchor = 1.1
    k_spring = 0.3
    return CoupledPendulums(mass, L_rod, d_anchor, k_spring)
