from abc import ABC, abstractmethod
import string
import subprocess
import ctypes
import hashlib
import numpy as np


class TimeIntegrator(ABC):
    def __init__(self, dynamical_system, dt):
        """Abstract base class for a single step traditional time integrator

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        """
        self.dynamical_system = dynamical_system
        self.dt = dt
        self.x = np.zeros(dynamical_system.dim)
        self.p = np.zeros(dynamical_system.dim)
        self.dHx = np.zeros(dynamical_system.dim)
        self.dHp = np.zeros(dynamical_system.dim)
        self.label = None

    def set_state(self, x, p):
        """Set the current state of the integrator to a specified
        position and momentum.

        :arg x: New position vector
        :arg p: New momentum vector
        """
        self.x[:] = x[:]
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
        return self.dynamical_system.energy(self.x, self.p)

    def _generate_timestepper_library(self, c_sourcecode):
        """Generate shared library from c source code

        The generated library will implement the timestepper. Returns ctypes wrapper
        which allows calling the function

          void timestepper(double* x, double* p, int nsteps) { ... }

        :arg c_sourcecode: C source code
        """
        # Check whether dynamical system has a C-code snippet for updating the acceleration
        self.fast_code = self.dynamical_system.dHx_update_code is not None
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
                DHX_UPDATE_CODE=self.dynamical_system.dHx_update_code,
                DHP_UPDATE_CODE=self.dynamical_system.dHp_update_code,
                DH_HEADER_CODE=header,
                DH_PREAMBLE_CODE=preamble,
            )
            sha = hashlib.md5()
            sha.update(c_substituted_sourcecode.encode())
            filestem = "./timestepper_" + sha.hexdigest()
            so_file = filestem + ".so"
            source_file = filestem + ".c"
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

        x_j^{(t+dt)} = x_j^{(t)} + dt*dH/dp_j (x^{(t)},p^{(t)})
        p_j^{(t+dt)} = p_j^{(t)} - dt*dH/dx_j (x^{(t)},p^{(t)})

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        """
        super().__init__(dynamical_system, dt)
        self.label = "ForwardEuler"

    def integrate(self, n_steps):
        """Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        """

        for _ in range(n_steps):
            # Compute forces
            self.dynamical_system.compute_dHx(self.x, self.p, self.dHx)
            self.dynamical_system.compute_dHp(self.x, self.p, self.dHp)
            # Update position and momentum
            self.x[:] += self.dt * self.dHp[:]
            self.p[:] -= self.dt * self.dHx[:]


class VerletIntegrator(TimeIntegrator):
    """Verlet integrator for separable Hamiltonians

    Performs the following update in every timestep:

        p_j^{(t+dt/2)} = p_j^{(t)} - dt/2 * dH/dx(x^{(t)})
        x_j^{(t+dt)}   = x_j^{(t)} + dt * dH/dp(p^{(t+dt/2)})
        p_j^{(t+dt)}   = p_j^{(t+dt/2)} - dt/2 * dH/dx(x^{(t+dt)})
    """

    def __init__(self, dynamical_system, dt):
        """Construct new instance

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        """
        super().__init__(dynamical_system, dt)
        self.label = "Verlet"
        c_sourcecode = """
        $DH_HEADER_CODE
        void timestepper(double* x, double* p, int nsteps) {
            double dHx[$DIM];
            double dHp[$DIM];
            $DH_PREAMBLE_CODE
            for (int k=0;k<nsteps;++k) {
                $DHX_UPDATE_CODE
                for (int j=0;j<$DIM;++j) {
                    p[j] -= 0.5*($DT)*dHx[j];
                }
                $DHP_UPDATE_CODE
                for (int j=0;j<$DIM;++j) {
                    x[j] += ($DT)*dHp[j];
                }
                $DHX_UPDATE_CODE
                for (int j=0;j<$DIM;++j) {
                    p[j] -= 0.5*($DT)*dHx[j];
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
            self.timestepper_library(self.x, self.p, n_steps)
        else:
            for _ in range(n_steps):
                self.dynamical_system.compute_dHx(self.x, self.p, self.dHx)
                self.p[:] -= 0.5 * self.dt * self.dHx[:]
                self.dynamical_system.compute_dHp(self.x, self.p, self.dHp)
                self.x[:] += self.dt * self.dHp[:]
                self.dynamical_system.compute_dHx(self.x, self.p, self.dHx)
                self.p[:] -= 0.5 * self.dt * self.dHx[:]


class RK4Integrator(TimeIntegrator):
    def __init__(self, dynamical_system, dt):
        """RK4 integrator given by

        (k_{1,x})_j = + dH/dp_j ( x^{(t)}, p^{(t)} )
        (k_{1,p})_j = - dH/dx_j ( x^{(t)}, p^{(t)} )

        (k_{2,x})_j = + dH/dp_j ( x^{(t)} + dt/2*k_{1,x}, p^{(t)} + dt/2*k_{1,p} )
        (k_{2,p})_j = - dH/dx_j ( x^{(t)} + dt/2*k_{1,x}, p^{(t)} + dt/2*k_{1,p} )

        (k_{3,x})_j = + dH/dp_j ( x^{(t)} + dt/2*k_{2,x}, p^{(t)} + dt/2*k_{2,p} )
        (k_{3,p})_j = - dH/dx_j ( x^{(t)} + dt/2*k_{2,x}, p^{(t)} + dt/2*k_{2,p} )

        (k_{4,x})_j = + dH/dp_j ( x^{(t)} + dt*k_{3,x}, p^{(t)} + dt*k_{3,p} )
        (k_{4,p})_j = - dH/dx_j ( x^{(t)} + dt*k_{3,x}, p^{(t)} + dt*k_{3,p} )

        x^{(t+dt)} = x^{(t)} + dt/6*( k_{1,x} + 2*k_{2,x} + 2*k_{3,x} + k_{4,x} )
        p^{(t+dt)} = p^{(t)} + dt/6*( k_{1,p} + 2*k_{2,p} + 2*k_{3,p} + k_{4,p} )

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        """
        super().__init__(dynamical_system, dt)
        self.label = "RK4"
        # temporary fields
        self.k1x = np.zeros(self.dynamical_system.dim)
        self.k1p = np.zeros(self.dynamical_system.dim)
        self.k2x = np.zeros(self.dynamical_system.dim)
        self.k2p = np.zeros(self.dynamical_system.dim)
        self.k3x = np.zeros(self.dynamical_system.dim)
        self.k3p = np.zeros(self.dynamical_system.dim)
        self.k4x = np.zeros(self.dynamical_system.dim)
        self.k4p = np.zeros(self.dynamical_system.dim)
        c_sourcecode = """
            $DH_HEADER_CODE
            void timestepper(double* x, double* p, int nsteps) {
                double dHx[$DIM];
                double dHp[$DIM];
                double k1x[$DIM];
                double k1p[$DIM];
                double k2x[$DIM];
                double k2p[$DIM];
                double k3x[$DIM];
                double k3p[$DIM];
                double k4x[$DIM];
                double k4p[$DIM];
                double xt[$DIM];
                double pt[$DIM];
                $DH_PREAMBLE_CODE
                for (int k=0;k<nsteps;++k) {
                    // *** Stage 1 *** compute k1
                    $DHX_UPDATE_CODE
                    $DHP_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        xt[j] = x[j];
                        pt[j] = p[j];
                        k1x[j] = + dHp[j];
                        k1p[j] = - dHx[j];
                        x[j] += 0.5*($DT)*k1x[j];
                        p[j] += 0.5*($DT)*k1p[j];
                    }
                    // *** Stage 2 *** compute k2
                    $DHX_UPDATE_CODE
                    $DHP_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        k2x[j] = + dHp[j];
                        k2p[j] = - dHx[j];
                        x[j] = xt[j] + 0.5*($DT)*k2x[j];
                        p[j] = pt[j] + 0.5*($DT)*k2p[j];
                    }
                    // *** Stage 3 *** compute k3
                    $DHX_UPDATE_CODE
                    $DHP_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        k3x[j] = + dHp[j];
                        k3p[j] = - dHx[j];
                        x[j] = xt[j] + ($DT)*k3x[j];
                        p[j] = pt[j] + ($DT)*k3p[j];
                    }
                    // *** Stage 4 *** compute k4
                    $DHX_UPDATE_CODE
                    $DHP_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        k4x[j] = + dHp[j];
                        k4p[j] = - dHx[j];
                    }
                    // *** Final stage *** combine k's to compute x^{(t+dt)} and p^{(t+dt)}
                    for (int j=0;j<$DIM;++j) {
                        x[j] = xt[j] + ($DT)/6.*(k1x[j]+2.*k2x[j]+2.*k3x[j]+k4x[j]);
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
            self.timestepper_library(self.x, self.p, n_steps)
        else:
            xt = np.zeros(self.dynamical_system.dim)
            pt = np.zeros(self.dynamical_system.dim)
            for _ in range(n_steps):
                xt[:] = self.x[:]
                pt[:] = self.p[:]
                # Stage 1: compute k1
                self.dynamical_system.compute_dHx(self.x, self.p, self.dHx)
                self.dynamical_system.compute_dHp(self.x, self.p, self.dHp)
                self.k1x[:] = +self.dHp[:]
                self.k1p[:] = -self.dHx[:]
                # Stage 2: compute k2
                self.x[:] += 0.5 * self.dt * self.k1x[:]
                self.p[:] += 0.5 * self.dt * self.k1p[:]
                self.dynamical_system.compute_dHx(self.x, self.p, self.dHx)
                self.dynamical_system.compute_dHp(self.x, self.p, self.dHp)
                # Stage 3: compute k3
                self.k2x[:] = +self.dHp[:]
                self.k2p[:] = -self.dHx[:]
                self.x[:] = xt[:] + 0.5 * self.dt * self.k2x[:]
                self.p[:] = pt[:] + 0.5 * self.dt * self.k2p[:]
                self.dynamical_system.compute_dHx(self.x, self.p, self.dHx)
                self.dynamical_system.compute_dHp(self.x, self.p, self.dHp)
                self.k3x[:] = +self.dHp[:]
                self.k3p[:] = -self.dHx[:]
                # Stage 4: compute k4
                self.x[:] = xt[:] + self.dt * self.k3x[:]
                self.p[:] = pt[:] + self.dt * self.k3p[:]
                self.dynamical_system.compute_dHx(self.x, self.p, self.dHx)
                self.dynamical_system.compute_dHp(self.x, self.p, self.dHp)
                self.k4x[:] = +self.dHp[:]
                self.k4p[:] = -self.dHx[:]
                # Final stage: combine k's
                self.x[:] = xt[:] + self.dt / 6.0 * (
                    self.k1x[:] + 2.0 * self.k2x[:] + 2.0 * self.k3x[:] + self.k4x[:]
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
        self.x[:], self.p[:] = self.dynamical_system.forward_map(
            self.x[:], self.p[:], n_steps * self.dt
        )
