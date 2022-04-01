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
        self.v = np.zeros(dynamical_system.dim)
        self.force = np.zeros(dynamical_system.dim)
        self.label = None

    def set_state(self, x, v):
        """Set the current state of the integrator to a specified
        position and velocity.

        :arg x: New position vector
        :arg v: New velocity vector
        """
        self.x[:] = x[:]
        self.v[:] = v[:]
        self.dynamical_system.compute_scaled_force(self.x, self.v, self.force)

    @abstractmethod
    def integrate(self, n_steps):
        """Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        """

    def energy(self):
        """Return the energy of the underlying dynamical system for
        the current position and velocity"""
        return self.dynamical_system.energy(self.x, self.v)


class ForwardEulerIntegrator(TimeIntegrator):
    def __init__(self, dynamical_system, dt):
        """Forward Euler integrator given by

        x_j^{(t+dt)} = x_j^{(t)} + dt*v_j
        v_j^{(t+dt)} = v_j^{(t)} + dt*F_j(x^{(t)})/m_j

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
            self.x[:] += self.dt * self.v[:]
            self.dynamical_system.apply_constraints(self.x)
            self.v[:] += self.dt * self.force[:]
            # Compute force at next timestep
            self.dynamical_system.compute_scaled_force(self.x, self.v, self.force)


class VerletIntegrator(TimeIntegrator):
    def __init__(self, dynamical_system, dt):
        """Verlet integrator given by

        x_j^{(t+dt)} = x_j^{(t)} + dt*v_j + dt^2/2*F_j(x^{(t)})/m_j
        v_j^{(t+dt)} = v_j^{(t)} + dt^2/2*(F_j(x^{(t)})/m_j+F_j(x^{(t+dt)})/m_j)

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        """
        super().__init__(dynamical_system, dt)
        self.label = "Verlet"
        # Check whether dynamical system has a C-code snippet for updating the acceleration
        self.fast_code = self.dynamical_system.acceleration_update_code is not None
        # If this is the case, auto-generate fast C code for the Velocity Verlet update
        if self.fast_code:
            if self.dynamical_system.acceleration_preamble_code:
                preamble = self.dynamical_system.acceleration_preamble_code
            else:
                preamble = ""
            if self.dynamical_system.acceleration_header_code:
                header = self.dynamical_system.acceleration_header_code
            else:
                header = ""
            c_sourcecode = string.Template(
                """
            $ACCELERATION_HEADER_CODE
            void velocity_verlet(double* x, double* v, int nsteps) {
                double a[$DIM];
                $ACCELERATION_PREAMBLE_CODE
                for (int k=0;k<nsteps;++k) {
                    for (int j=0;j<$DIM;++j) a[j] = 0;
                    $ACCELERATION_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        x[j] += $DT*v[j] + 0.5*$DT*$DT*a[j];
                    }
                    $ACCELERATION_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        v[j] += 0.5*$DT*a[j];
                    }
                }
            }
            """
            ).substitute(
                DIM=self.dynamical_system.dim,
                DT=self.dt,
                ACCELERATION_UPDATE_CODE=self.dynamical_system.acceleration_update_code,
                ACCELERATION_HEADER_CODE=header,
                ACCELERATION_PREAMBLE_CODE=preamble,
            )
            sha = hashlib.md5()
            sha.update(c_sourcecode.encode())
            filestem = "./velocity_verlet_" + sha.hexdigest()
            so_file = filestem + ".so"
            source_file = filestem + ".c"
            with open(source_file, "w", encoding="utf8") as f:
                print(c_sourcecode, file=f)
            # Compile source code (might have to adapt for different compiler)
            subprocess.run(
                ["gcc", "-fPIC", "-shared", "-o", so_file, source_file], check=True
            )
            self.c_velocity_verlet = ctypes.CDLL(so_file).velocity_verlet
            self.c_velocity_verlet.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                np.ctypeslib.c_intp,
            ]

    def integrate(self, n_steps):
        """Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        """
        if self.fast_code:
            self.c_velocity_verlet(self.x, self.v, n_steps)
        else:
            for _ in range(n_steps):
                self.x[:] += self.dt * self.v[:] + 0.5 * self.dt**2 * self.force[:]
                self.dynamical_system.apply_constraints(self.x)
                self.v[:] += 0.5 * self.dt * self.force[:]
                self.dynamical_system.compute_scaled_force(self.x, self.v, self.force)
                self.v[:] += 0.5 * self.dt * self.force[:]


class RK4Integrator(TimeIntegrator):
    def __init__(self, dynamical_system, dt):
        """RK4 integrator given by

        (k_{1,v})_j = v_j^{(t)}
        (k_{1,x})_j = F_j(x^{(t)},v^{(t)})/m_j

        (k_{2,v})_j = v_j^{(t)} + dt/2*(k_{1,v})_j
        (k_{2,x})_j = F_j(x^{(t)}) + dt/2*k_{1,x}, v^{(t)} + dt/2*k_{1,v})/m_j

        (k_{3,v})_j = v_j^{(t)} + dt/2*(k_{2,v})_j
        (k_{3,x})_j = F_j(x^{(t)}) + dt/2*k_{2,x}, v^{(t)} + dt/2*k_{2,v})/m_j

        (k_{4,v})_j = v_j^{(t)} + dt*(k_{3,v})_j
        (k_{4,x})_j = F_j(x^{(t)}) + dt*k_{3,x}, v^{(t)} + dt*k_{3,v})/m_j

        x^{(t+dt)} = x^{(t)} + dt/6*(k_{1,x}+2*k_{2,x}+2*k_{3,x}+k_{4,x})
        v^{(t+dt)} = v^{(t)} + dt/6*(k_{1,v}+2*k_{2,v}+2*k_{3,v}+k_{4,v})

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        """
        super().__init__(dynamical_system, dt)
        self.label = "RK4"
        # temporary fields
        self.k1x = np.zeros(self.dynamical_system.dim)
        self.k1v = np.zeros(self.dynamical_system.dim)
        self.k2x = np.zeros(self.dynamical_system.dim)
        self.k2v = np.zeros(self.dynamical_system.dim)
        self.k3x = np.zeros(self.dynamical_system.dim)
        self.k3v = np.zeros(self.dynamical_system.dim)
        self.k4x = np.zeros(self.dynamical_system.dim)
        self.k4v = np.zeros(self.dynamical_system.dim)
        # Check whether dynamical system has a C-code snippet for updating the acceleration
        self.fast_code = self.dynamical_system.acceleration_update_code is not None
        # If this is the case, auto-generate fast C code for the Velocity Verlet update
        if self.fast_code:
            if self.dynamical_system.acceleration_preamble_code:
                preamble = self.dynamical_system.acceleration_preamble_code
            else:
                preamble = ""
            if self.dynamical_system.acceleration_header_code:
                header = self.dynamical_system.acceleration_header_code
            else:
                header = ""
            c_sourcecode = string.Template(
                """
            $ACCELERATION_HEADER_CODE
            void rk4(double* x, double* v, int nsteps) {
                double a[$DIM];
                double k1x[$DIM];
                double k1v[$DIM];
                double k2x[$DIM];
                double k2v[$DIM];
                double k3x[$DIM];
                double k3v[$DIM];
                double k4x[$DIM];
                double k4v[$DIM];
                double xt[$DIM];
                double vt[$DIM];
                $ACCELERATION_PREAMBLE_CODE
                for (int k=0;k<nsteps;++k) {
                    // *** Stage 1 *** compute k1
                    for (int j=0;j<$DIM;++j) a[j] = 0;
                    $ACCELERATION_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        xt[j] = x[j];
                        vt[j] = v[j];
                        k1x[j] = v[j];
                        k1v[j] = a[j];
                        x[j] += 0.5*$DT*k1x[j];
                        v[j] += 0.5*$DT*k1v[j];
                        a[j] = 0;
                    }
                    // *** Stage 2 *** compute k2
                    $ACCELERATION_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        k2x[j] = v[j];
                        k2v[j] = a[j];
                        x[j] = xt[j] + 0.5*$DT*k2x[j];
                        v[j] = vt[j] + 0.5*$DT*k2v[j];
                        a[j] = 0;
                    }
                    // *** Stage 3 *** compute k3
                    $ACCELERATION_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        k3x[j] = v[j];
                        k3v[j] = a[j];
                        x[j] = xt[j] + $DT*k3x[j];
                        v[j] = vt[j] + $DT*k3v[j];
                        a[j] = 0;
                    }
                    // *** Stage 4 *** compute k4
                    $ACCELERATION_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        k4x[j] = v[j];
                        k4v[j] = a[j];
                    }
                    // *** Final stage *** combine k's to compute x^{(t+dt)}
                    for (int j=0;j<$DIM;++j) {
                        x[j] = xt[j] + $DT/6.*(k1x[j]+2.*k2x[j]+2.*k3x[j]+k4x[j]);
                        v[j] = vt[j] + $DT/6.*(k1v[j]+2.*k2v[j]+2.*k3v[j]+k4v[j]);
                    }
                }
            }
            """
            ).substitute(
                DIM=self.dynamical_system.dim,
                DT=self.dt,
                ACCELERATION_UPDATE_CODE=self.dynamical_system.acceleration_update_code,
                ACCELERATION_HEADER_CODE=header,
                ACCELERATION_PREAMBLE_CODE=preamble,
            )
            sha = hashlib.md5()
            sha.update(c_sourcecode.encode())
            filestem = "./rk4_" + sha.hexdigest()
            so_file = filestem + ".so"
            source_file = filestem + ".c"
            with open(source_file, "w", encoding="utf8") as f:
                print(c_sourcecode, file=f)
            # Compile source code (might have to adapt for different compiler)
            subprocess.run(
                ["gcc", "-fPIC", "-shared", "-o", so_file, source_file], check=True
            )
            self.c_rk4 = ctypes.CDLL(so_file).rk4
            self.c_rk4.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                np.ctypeslib.c_intp,
            ]

    def integrate(self, n_steps):
        """Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        """
        if self.fast_code:
            self.c_rk4(self.x, self.v, n_steps)
        else:
            xt = np.zeros(self.dynamical_system.dim)
            vt = np.zeros(self.dynamical_system.dim)
            for _ in range(n_steps):
                xt[:] = self.x[:]
                vt[:] = self.v[:]
                # Stage 1: compute k1
                self.k1x[:] = self.v[:]
                self.k1v[:] = self.force[:]
                # Stage 2: compute k2
                self.x[:] = xt[:] + 0.5 * self.dt * self.k1x[:]
                self.v[:] = vt[:] + 0.5 * self.dt * self.k1v[:]
                self.dynamical_system.apply_constraints(self.x)
                self.dynamical_system.compute_scaled_force(self.x, self.v, self.force)
                self.k2x[:] = self.v[:]
                self.k2v[:] = self.force[:]
                # Stage 3: compute k3
                self.x[:] = xt[:] + 0.5 * self.dt * self.k2x[:]
                self.v[:] = vt[:] + 0.5 * self.dt * self.k2v[:]
                self.dynamical_system.apply_constraints(self.x)
                self.dynamical_system.compute_scaled_force(self.x, self.v, self.force)
                self.k3x[:] = self.v[:]
                self.k3v[:] = self.force[:]
                # Stage 4: compute k4
                self.x[:] = xt[:] + self.dt * self.k3x[:]
                self.v[:] = vt[:] + self.dt * self.k3v[:]
                self.dynamical_system.apply_constraints(self.x)
                self.dynamical_system.compute_scaled_force(self.x, self.v, self.force)
                self.k4x[:] = self.v[:]
                self.k4v[:] = self.force[:]
                # Final stage: combine k's
                self.x[:] = xt[:] + self.dt / 6.0 * (
                    self.k1x[:] + 2.0 * self.k2x[:] + 2.0 * self.k3x[:] + self.k4x[:]
                )
                self.v[:] = vt[:] + self.dt / 6.0 * (
                    self.k1v[:] + 2.0 * self.k2v[:] + 2.0 * self.k3v[:] + self.k4v[:]
                )
                self.dynamical_system.apply_constraints(self.x)
                self.dynamical_system.compute_scaled_force(self.x, self.v, self.force)


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
        self.x[:], self.v[:] = self.dynamical_system.forward_map(
            self.x[:], self.v[:], n_steps * self.dt
        )
