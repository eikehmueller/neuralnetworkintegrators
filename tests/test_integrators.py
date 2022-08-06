import os
import pytest
import numpy as np
from matplotlib import pyplot as plt
import fixtures
from fixtures import *  # pylint: disable=import-error,wildcard-import
from dynamical_system import *  # pylint: disable=import-error,wildcard-import
from time_integrator import *  # pylint: disable=import-error,wildcard-import


@pytest.mark.parametrize(
    "TimestepperClass",
    [
        ForwardEulerIntegrator,
        VerletIntegrator,
        RK4Integrator,
        StrangSplittingIntegrator,
    ],
)
def test_time_integrator_fastcode(harmonic_oscillator, TimestepperClass):
    """Check that time integrators give the same result for Python and C code

    The test is parametrised over the different timestepper classes
    """
    tolerance = 1.0e-12
    dt = 0.1
    dynamical_system = harmonic_oscillator
    dim = dynamical_system.dim
    time_stepper = TimestepperClass(dynamical_system, dt)
    qp = np.zeros((2, 2 * dim))
    n_step = 10
    q0 = np.ones(dim)
    p0 = np.ones(dim)
    for j, fast_code in enumerate((True, False)):
        time_stepper.set_state(q0, p0)
        time_stepper.fast_code = fast_code
        time_stepper.integrate(n_step)
        qp[j, :dim] = time_stepper.q[:]
        qp[j, dim:] = time_stepper.p[:]
    diff = np.linalg.norm(qp[0, :] - qp[1, :])
    assert diff < tolerance


@pytest.mark.parametrize(
    "dynamical_system_name",
    [
        "harmonic_oscillator",
        "coupled_harmonic_oscillators",
        "double_pendulum",
        "coupled_pendulums",
    ],
)
def test_dynamical_system_fastcode(dynamical_system_name, request):
    """Check that dynamical systems give same results for Python and C code

    The test is parametrised over the different dynamical systems
    """
    tolerance = 1.0e-12
    dt = 0.1
    dynamical_system = request.getfixturevalue(dynamical_system_name)
    dim = dynamical_system.dim
    time_stepper = RK4Integrator(dynamical_system, dt)
    qp = np.zeros((2, 2 * dim))
    n_step = 2
    q0 = np.ones(dim)
    p0 = np.ones(dim)
    for j, fast_code in enumerate((True, False)):
        time_stepper.set_state(q0, p0)
        time_stepper.fast_code = fast_code
        time_stepper.integrate(n_step)
        qp[j, :dim] = time_stepper.q[:]
        qp[j, dim:] = time_stepper.p[:]
    diff = np.linalg.norm(qp[0, :] - qp[1, :])
    assert diff < tolerance


def convergence(Integrator, dynamical_system, output_dir="./convergence_plots/"):
    """Generate plot of error vs timestep size and work out order of integrator

    Returns the order of convergence

    :arg Integrator: timestepper class to use
    :arg dynamical_system: dynamical system to integrate
    :arg output_dir: subdirectory to store the tests in
    """
    # Final time
    t_final = 1.0
    dim = dynamical_system.dim
    # Set initial conditions
    q0, p0 = np.ones(dim), np.zeros(dim)
    # Compute exact solution
    try:
        exact_time_integrator = ExactIntegrator(dynamical_system, t_final)
        exact_time_integrator.set_state(q0, p0)
        exact_time_integrator.integrate(1)
    except NotImplementedError:
        n_steps_exact = 4096
        exact_time_integrator = RK4Integrator(dynamical_system, t_final / n_steps_exact)
        exact_time_integrator.set_state(q0, p0)
        exact_time_integrator.integrate(n_steps_exact)
    # Numbers of timesteps used for integration
    n_steps_list = 2 ** (np.arange(2, 11, 1))
    dt_list = []
    error_list = []
    diff = np.zeros(2 * dim)
    for _, n_steps in enumerate(n_steps_list):
        dt = t_final / n_steps
        time_integrator = Integrator(dynamical_system, dt)
        time_integrator.set_state(q0, p0)
        time_integrator.integrate(n_steps)
        diff[:dim] = time_integrator.q[:] - exact_time_integrator.q[:]
        diff[dim:] = time_integrator.p[:] - exact_time_integrator.p[:]
        # Do not include error that are smaller than 1E-12
        error = np.linalg.norm(diff)
        if error > 1.0e-12:
            dt_list.append(dt)
            error_list.append(error)
    dt_list = np.array(dt_list)
    error_list = np.array(error_list)
    log_dt = np.log2(dt_list)
    log_error = np.log2(error_list)
    n_fit = 4
    # Fit a polynomial to the logarithm of the error to work out the order of the method
    order, log_C0 = np.polyfit(log_dt[-n_fit:], log_error[-n_fit:], deg=1)
    plt.clf()
    plt.plot(
        dt_list[:],
        error_list[:],
        linewidth=2,
        color="blue",
        markersize=8,
        markeredgewidth=2,
        markerfacecolor="white",
        marker="o",
        label="data",
    )
    plt.plot(
        dt_list[-n_fit:],
        error_list[-n_fit:],
        linewidth=2,
        color="blue",
        markersize=8,
        markeredgewidth=2,
        markerfacecolor="blue",
        marker="o",
        label="data",
    )
    H = dt_list[:]
    C0 = 2**log_C0
    plt.plot(
        H,
        C0 * H**order,
        linewidth=2,
        color="red",
        label=r"fit $\propto h^{" + f"{order:4.2f}" + "}$",
    )
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"timestep size $\Delta t$")
    ax.set_ylabel("error $||qp-qp_{exact}||$")
    plt.legend(loc="lower right")
    integrator_label = Integrator.__name__
    dynamical_system_label = dynamical_system.__class__.__name__
    plt.title(f"integrator: {integrator_label}")
    plt.suptitle(f"dynamical system: {dynamical_system_label}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(
        output_dir
        + "convergence_"
        + integrator_label
        + "_"
        + dynamical_system_label
        + ".pdf",
        bbox_inches="tight",
    )
    return order


@pytest.mark.parametrize(
    "dynamical_system_name",
    [
        "harmonic_oscillator",
        "coupled_harmonic_oscillators",
        "double_pendulum",
        "coupled_pendulums",
    ],
)
@pytest.mark.parametrize(
    "TimestepperClass,expected",
    [
        (ForwardEulerIntegrator, 1),
        (VerletIntegrator, 2),
        (RK4Integrator, 4),
        (StrangSplittingIntegrator, 2),
    ],
)
def test_convergence(dynamical_system_name, TimestepperClass, expected, request):
    """Check that timestepper converges with the expected order

    The test is parametrised over the different timestepper classes and a range of
    dynamical systems
    """
    dynamical_system = request.getfixturevalue(dynamical_system_name)
    if (TimestepperClass is VerletIntegrator) and (not dynamical_system.separable):
        pytest.skip("VerletIntegrator only works for separable systems")
    order = convergence(TimestepperClass, dynamical_system)
    assert abs(order - expected) < 0.1
