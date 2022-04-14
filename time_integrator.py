from abc import ABC, abstractmethod
import string
import subprocess
import ctypes
import hashlib
import os
import numpy as np


class TimeIntegrator(ABC):
    def __init__(self, dynamical_system, dt):
        """Abstract base class for a single step traditional time integrator

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        """
        self.dynamical_system = dynamical_system
        self.dt = dt
        self.q = np.zeros(dynamical_system.dim)
        self.p = np.zeros(dynamical_system.dim)
        self.dHq = np.zeros(dynamical_system.dim)
        self.dHp = np.zeros(dynamical_system.dim)
        self.label = None
        # Check whether dynamical system has a C-code snippet for updating the acceleration
        self.fast_code = self.dynamical_system.dHq_update_code is not None

    def set_state(self, q, p):
        """Set the current state of the integrator to a specified
        position and momentum.

        :arg q: New position vector
        :arg p: New momentum vector
        """
        self.q[:] = q[:]
        self.p[:] = p[:]

    @abstractmethod
    def integrate(self, n_steps):
        """Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        """

    def energy(self):
        """Return the energy of the underlying dynamical system for
        the current position and velocity"""
        return self.dynamical_system.energy(self.q, self.p)

    def _generate_timestepper_library(self, c_sourcecode):
        """Generate shared library from c source code

        The generated library will implement the timestepper. Returns ctypes wrapper
        which allows calling the function

          void timestepper(double* q, double* p, int nsteps) { ... }

        :arg c_sourcecode: C source code
        """
        # If this is the case, auto-generate fast C code for the Velocity Verlet update

        if self.fast_code:
            if self.dynamical_system.dH_preamble_code:
                preamble = self.dynamical_system.dH_preamble_code
            else:
                preamble = ""
            if self.dynamical_system.dH_header_code:
                header = self.dynamical_system.dH_header_code
            else:
                header = ""
            c_substituted_sourcecode = string.Template(c_sourcecode).substitute(
                DIM=self.dynamical_system.dim,
                DT=self.dt,
                DHQ_UPDATE_CODE=self.dynamical_system.dHq_update_code,
                DHP_UPDATE_CODE=self.dynamical_system.dHp_update_code,
                DH_HEADER_CODE=header,
                DH_PREAMBLE_CODE=preamble,
            )
            sha = hashlib.md5()
            sha.update(c_substituted_sourcecode.encode())
            directory = "./generated_code/"
            if not os.stat(directory):
                os.mkdir(directory)
            filestem = "./timestepper_" + sha.hexdigest()
            so_file = directory + "/" + filestem + ".so"
            source_file = directory + "/" + filestem + ".c"
            with open(source_file, "w", encoding="utf8") as f:
                print(c_substituted_sourcecode, file=f)
            # Compile source code (might have to adapt for different compiler)
            subprocess.run(
                ["gcc", "-fPIC", "-shared", "-O3", "-o", so_file, source_file],
                check=True,
            )
            timestepper_lib = ctypes.CDLL(so_file).timestepper
            timestepper_lib.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                np.ctypeslib.c_intp,
            ]
            return timestepper_lib
        else:
            return None


class ForwardEulerIntegrator(TimeIntegrator):
    def __init__(self, dynamical_system, dt):
        """Forward Euler integrator given by

        q_j^{(t+dt)} = q_j^{(t)} + dt*dH/dp_j (q^{(t)},p^{(t)})
        p_j^{(t+dt)} = p_j^{(t)} - dt*dH/dq_j (q^{(t)},p^{(t)})

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        """
        super().__init__(dynamical_system, dt)
        self.label = "ForwardEuler"
        c_sourcecode = """
        $DH_HEADER_CODE
        void timestepper(double* q, double* p, int nsteps) {
            double dHq[$DIM];
            double dHp[$DIM];
            $DH_PREAMBLE_CODE
            for (int k=0;k<nsteps;++k) {
                $DHQ_UPDATE_CODE
                $DHP_UPDATE_CODE
                for (int j=0;j<$DIM;++j) {
                    q[j] += ($DT)*dHp[j];
                    p[j] -= ($DT)*dHq[j];
                }
            }
        }
        """
        self.timestepper_library = self._generate_timestepper_library(c_sourcecode)

    def integrate(self, n_steps):
        """Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        """
        if self.fast_code:
            self.timestepper_library(self.q, self.p, n_steps)
        else:
            for _ in range(n_steps):
                # Compute forces
                self.dynamical_system.compute_dHq(self.q, self.p, self.dHq)
                self.dynamical_system.compute_dHp(self.q, self.p, self.dHp)
                # Update position and momentum
                self.q[:] += self.dt * self.dHp[:]
                self.p[:] -= self.dt * self.dHq[:]


class VerletIntegrator(TimeIntegrator):
    """Verlet integrator for separable Hamiltonians

    Performs the following update in every timestep:

        p_j^{(t+dt/2)} = p_j^{(t)} - dt/2 * dH/dq(q^{(t)})
        q_j^{(t+dt)}   = q_j^{(t)} + dt * dH/dp(p^{(t+dt/2)})
        p_j^{(t+dt)}   = p_j^{(t+dt/2)} - dt/2 * dH/dq(q^{(t+dt)})
    """

    def __init__(self, dynamical_system, dt):
        """Construct new instance

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        """
        super().__init__(dynamical_system, dt)
        if not self.dynamical_system.separable:
            raise Exception(
                "Verlet integrator only support separable Hamiltonian systems"
            )
        self.label = "Verlet"
        c_sourcecode = """
        $DH_HEADER_CODE
        void timestepper(double* q, double* p, int nsteps) {
            double dHq[$DIM];
            double dHp[$DIM];
            $DH_PREAMBLE_CODE
            for (int k=0;k<nsteps;++k) {
                $DHQ_UPDATE_CODE
                for (int j=0;j<$DIM;++j) {
                    p[j] -= 0.5*($DT)*dHq[j];
                }
                $DHP_UPDATE_CODE
                for (int j=0;j<$DIM;++j) {
                    q[j] += ($DT)*dHp[j];
                }
                $DHQ_UPDATE_CODE
                for (int j=0;j<$DIM;++j) {
                    p[j] -= 0.5*($DT)*dHq[j];
                }
            }
        }
        """
        self.timestepper_library = self._generate_timestepper_library(c_sourcecode)

    def integrate(self, n_steps):
        """Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        """
        if self.fast_code:
            self.timestepper_library(self.q, self.p, n_steps)
        else:
            for _ in range(n_steps):
                self.dynamical_system.compute_dHq(self.q, self.p, self.dHq)
                self.p[:] -= 0.5 * self.dt * self.dHq[:]
                self.dynamical_system.compute_dHp(self.q, self.p, self.dHp)
                self.q[:] += self.dt * self.dHp[:]
                self.dynamical_system.compute_dHq(self.q, self.p, self.dHq)
                self.p[:] -= 0.5 * self.dt * self.dHq[:]


class RK4Integrator(TimeIntegrator):
    def __init__(self, dynamical_system, dt):
        """RK4 integrator given by

        (k_{1,q})_j = + dH/dp_j ( q^{(t)}, p^{(t)} )
        (k_{1,p})_j = - dH/dq_j ( q^{(t)}, p^{(t)} )

        (k_{2,q})_j = + dH/dp_j ( q^{(t)} + dt/2*k_{1,q}, p^{(t)} + dt/2*k_{1,p} )
        (k_{2,p})_j = - dH/dq_j ( q^{(t)} + dt/2*k_{1,q}, p^{(t)} + dt/2*k_{1,p} )

        (k_{3,q})_j = + dH/dp_j ( q^{(t)} + dt/2*k_{2,q}, p^{(t)} + dt/2*k_{2,p} )
        (k_{3,p})_j = - dH/dq_j ( q^{(t)} + dt/2*k_{2,q}, p^{(t)} + dt/2*k_{2,p} )

        (k_{4,q})_j = + dH/dp_j ( q^{(t)} + dt*k_{3,q}, p^{(t)} + dt*k_{3,p} )
        (k_{4,p})_j = - dH/dq_j ( q^{(t)} + dt*k_{3,q}, p^{(t)} + dt*k_{3,p} )

        q^{(t+dt)} = q^{(t)} + dt/6*( k_{1,q} + 2*k_{2,q} + 2*k_{3,q} + k_{4,q} )
        p^{(t+dt)} = p^{(t)} + dt/6*( k_{1,p} + 2*k_{2,p} + 2*k_{3,p} + k_{4,p} )

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        """
        super().__init__(dynamical_system, dt)
        self.label = "RK4"
        # temporary fields
        self.k1q = np.zeros(self.dynamical_system.dim)
        self.k1p = np.zeros(self.dynamical_system.dim)
        self.k2q = np.zeros(self.dynamical_system.dim)
        self.k2p = np.zeros(self.dynamical_system.dim)
        self.k3q = np.zeros(self.dynamical_system.dim)
        self.k3p = np.zeros(self.dynamical_system.dim)
        self.k4q = np.zeros(self.dynamical_system.dim)
        self.k4p = np.zeros(self.dynamical_system.dim)
        c_sourcecode = """
            $DH_HEADER_CODE
            void timestepper(double* q, double* p, int nsteps) {
                double dHq[$DIM];
                double dHp[$DIM];
                double k1q[$DIM];
                double k1p[$DIM];
                double k2q[$DIM];
                double k2p[$DIM];
                double k3q[$DIM];
                double k3p[$DIM];
                double k4q[$DIM];
                double k4p[$DIM];
                double qt[$DIM];
                double pt[$DIM];
                $DH_PREAMBLE_CODE
                for (int k=0;k<nsteps;++k) {
                    // *** Stage 1 *** compute k1
                    $DHQ_UPDATE_CODE
                    $DHP_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        qt[j] = q[j];
                        pt[j] = p[j];
                        k1q[j] = + dHp[j];
                        k1p[j] = - dHq[j];
                        q[j] += 0.5*($DT)*k1q[j];
                        p[j] += 0.5*($DT)*k1p[j];
                    }
                    // *** Stage 2 *** compute k2
                    $DHQ_UPDATE_CODE
                    $DHP_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        k2q[j] = + dHp[j];
                        k2p[j] = - dHq[j];
                        q[j] = qt[j] + 0.5*($DT)*k2q[j];
                        p[j] = pt[j] + 0.5*($DT)*k2p[j];
                    }
                    // *** Stage 3 *** compute k3
                    $DHQ_UPDATE_CODE
                    $DHP_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        k3q[j] = + dHp[j];
                        k3p[j] = - dHq[j];
                        q[j] = qt[j] + ($DT)*k3q[j];
                        p[j] = pt[j] + ($DT)*k3p[j];
                    }
                    // *** Stage 4 *** compute k4
                    $DHQ_UPDATE_CODE
                    $DHP_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        k4q[j] = + dHp[j];
                        k4p[j] = - dHq[j];
                    }
                    // *** Final stage *** combine k's to compute q^{(t+dt)} and p^{(t+dt)}
                    for (int j=0;j<$DIM;++j) {
                        q[j] = qt[j] + ($DT)/6.*(k1q[j]+2.*k2q[j]+2.*k3q[j]+k4q[j]);
                        p[j] = pt[j] + ($DT)/6.*(k1p[j]+2.*k2p[j]+2.*k3p[j]+k4p[j]);
                    }
                }
            }
            """
        self.timestepper_library = self._generate_timestepper_library(c_sourcecode)

    def integrate(self, n_steps):
        """Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        """
        if self.fast_code:
            self.timestepper_library(self.q, self.p, n_steps)
        else:
            qt = np.zeros(self.dynamical_system.dim)
            pt = np.zeros(self.dynamical_system.dim)
            for _ in range(n_steps):
                qt[:] = self.q[:]
                pt[:] = self.p[:]
                # Stage 1: compute k1
                self.dynamical_system.compute_dHq(self.q, self.p, self.dHq)
                self.dynamical_system.compute_dHp(self.q, self.p, self.dHp)
                self.k1q[:] = +self.dHp[:]
                self.k1p[:] = -self.dHq[:]
                # Stage 2: compute k2
                self.q[:] += 0.5 * self.dt * self.k1q[:]
                self.p[:] += 0.5 * self.dt * self.k1p[:]
                self.dynamical_system.compute_dHq(self.q, self.p, self.dHq)
                self.dynamical_system.compute_dHp(self.q, self.p, self.dHp)
                # Stage 3: compute k3
                self.k2q[:] = +self.dHp[:]
                self.k2p[:] = -self.dHq[:]
                self.q[:] = qt[:] + 0.5 * self.dt * self.k2q[:]
                self.p[:] = pt[:] + 0.5 * self.dt * self.k2p[:]
                self.dynamical_system.compute_dHq(self.q, self.p, self.dHq)
                self.dynamical_system.compute_dHp(self.q, self.p, self.dHp)
                self.k3q[:] = +self.dHp[:]
                self.k3p[:] = -self.dHq[:]
                # Stage 4: compute k4
                self.q[:] = qt[:] + self.dt * self.k3q[:]
                self.p[:] = pt[:] + self.dt * self.k3p[:]
                self.dynamical_system.compute_dHq(self.q, self.p, self.dHq)
                self.dynamical_system.compute_dHp(self.q, self.p, self.dHp)
                self.k4q[:] = +self.dHp[:]
                self.k4p[:] = -self.dHq[:]
                # Final stage: combine k's
                self.q[:] = qt[:] + self.dt / 6.0 * (
                    self.k1q[:] + 2.0 * self.k2q[:] + 2.0 * self.k3q[:] + self.k4q[:]
                )
                self.p[:] = pt[:] + self.dt / 6.0 * (
                    self.k1p[:] + 2.0 * self.k2p[:] + 2.0 * self.k3p[:] + self.k4p[:]
                )


class ExactIntegrator(TimeIntegrator):
    def __init__(self, dynamical_system, dt):
        """Exact integrator

        Integrate the equations of motion exactly, if the dynamical system supports this.

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        """
        super().__init__(dynamical_system, dt)
        self.label = "Exact"

    def integrate(self, n_steps):
        """Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        """
        self.q[:], self.p[:] = self.dynamical_system.forward_map(
            self.q[:], self.p[:], n_steps * self.dt
        )
