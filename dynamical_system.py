from abc import ABC, abstractmethod
import ctypes
import subprocess
import string
from matplotlib import pyplot as plt
import numpy as np


class DynamicalSystem(ABC):
    """Abstract base class for a dynamical system

    Models a d-dimensional system of the following form:

      dx_j/dt = v_j
      dv_j/dt = F_j(x)/m_j

    where j = 0,1,2,...,d-1
    """

    def __init__(self, dim, mass):
        """Construct new instance, set dimension and mass

        :arg dim: Spatial dimension of dynamical system
        :arg mass: Mass of the system (can be a scalar in 1d or a list in heigher dimensions)
        """
        self.dim = dim
        self.mass = mass
        self.acceleration_header_code = None
        self.acceleration_preamble_code = None
        self.acceleration_update_code = None

    @abstractmethod
    def compute_acceleration(self, x, v, acceleration):
        """Store the acceleration (forces scaled by inverse masses) in the vector
        such that acceleration[j] = F_j(x)/m_j

        :arg x: Particle positions x (d-dimensional array)
        :arg v: Particle velocities x (d-dimensional array)
        :arg acceleration: Resulting acceleration vector (d-dimensional array)
        """

    @abstractmethod
    def set_random_state(self, x, v):
        """Set the position x and v to random values. This will be used
        during the training stage to pick a suitable set of initial values

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        """

    @abstractmethod
    def energy(self, x, v):
        """Return the total energy for given positions and velocities

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        """

    def apply_constraints(self, x):
        """Apply constraints, such as period boundary conditions

        :arg x: Positions (d-dimensional array)
        """

    def forward_map(self, x0, v0, t):
        """Exact forward map

        Compute position x(t) and velocity v(t), given initial position x(0) and velocity v(0).
        This will only be implemented if the specific dynamical system has an analytical solution

        :arg x0: initial position x(0)
        :arg v0: initial velocity v(0)
        :arg t: final time
        """
        raise NotImplementedError("Dynamical system has no exact solution.")


class HarmonicOscillator(DynamicalSystem):
    """One-dimensional harmonic oscillator described by the equations
    of motion

    dx_0/dt = v_0, dv_0/dt = -k/m_0*x_0
    """

    def __init__(self, mass, k_spring):
        """Construct new instance of harmonic oscillator class

        :arg mass: Particle mass
        :arg k_spring: Spring constant k
        """
        super().__init__(1, mass)
        self.k_spring = k_spring
        # C-code snipped for computing the acceleration update
        self.acceleration_update_code = f"a[0] += -({self.k_spring}/{self.mass})*x[0];"

    def compute_acceleration(self, x, v, acceleration):
        """Set the entry acceleration[0] of the acceleration vector to -k/m_0*x_0

        :arg x: Particle position x (1-dimensional array)
        :arg v: Particle velocities x (d-dimensional array)
        :arg acceleration: Resulting acceleration vector (1-dimensional array)
        """
        acceleration[0] = -self.k_spring / self.mass * x[0]

    def set_random_state(self, x, v):
        """Draw position and velocity from a normal distribution
        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        """
        x[0] = np.random.normal(0, 1)
        v[0] = np.random.normal(0, 1)

    def energy(self, x, v):
        """Compute total energy E = 1/2*m*v_0^2 + 1/2*k*x_0^2

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        """
        return 0.5 * self.mass * v[0] ** 2 + 0.5 * self.k_spring * x[0] ** 2

    def forward_map(self, x0, v0, t):
        """Exact forward map

        Compute position x(t) and velocity v(t), given initial position x(0) and velocity v(0).

        For this use:

        x(t) = x(0)*cos(omega*t) + omega*v(0)*sin(omega*t)
        v(t) = -x(0)/omega*sin(omega*t) + v(0)*cos(omega*t)

        with omegae = sqrt(k/m)

        :arg x0: initial position x(0)
        :arg v0: initial velocity v(0)
        :arg t: final time
        """
        omega = np.sqrt(self.k_spring / self.mass)
        cos_omegat = np.cos(omega * t)
        sin_omegat = np.sin(omega * t)
        x = np.array(x0[0] * cos_omegat + v0[0] / omega * sin_omegat)
        v = np.array(-x0[0] * omega * sin_omegat + v0[0] * cos_omegat)
        return x, v


class DoublePendulum(DynamicalSystem):
    """2-dimensional double pendulum described by the equations
    of motion

    dx_0/dt = v_0, dv_0/dt = [a1,a2] in scaled acceleration section
    """

    def __init__(self, mass, L1, L2, g=9.81):
        """Construct new instance of double pendulum class.

        :arg mass: Particle mass (list of length two)
        :arg g: gravitional acceleration constant
        :arg L1: length of first segment of double pendulum
        :arg L2: length of second segment of double pendulum
        """
        super().__init__(2, mass)
        self.g = g
        self.L1 = L1
        self.L2 = L2
        # C-code snipped for computing the acceleration update
        self.acceleration_header_code = """
        #include "math.h"
        """
        self.acceleration_preamble_code = """
        double cos_x0_x1;
        double sin_x0_x1;
        double sin_x0;
        double sin_x1;
        """
        self.acceleration_update_code = """
        cos_x0_x1 = cos(x[0]-x[1]);
        sin_x0_x1 = sin(x[0]-x[1]);
        sin_x0 = sin(x[0]);
        sin_x1 = sin(x[1]);
        a[0] += 1/({L1}*({mu} - cos_x0_x1*cos_x0_x1))
             * ({g}*(sin_x1*cos_x0_x1-{mu}*sin_x0)
             - ({L2}*v[1]*v[1] + {L1}*v[0]*v[0]*cos_x0_x1)*sin_x0_x1);
        a[1] += 1/({L2}*({mu} - cos_x0_x1*cos_x0_x1))
             * ({g}*{mu}*(sin_x0*cos_x0_x1-sin_x1)
             + ({L1}*{mu}*v[0]*v[0]+{L2}*v[1]*v[1]*cos_x0_x1)*sin_x0_x1);
        """.format(
            mu=1 + self.mass[0] + self.mass[1], L1=self.L1, L2=self.L2, g=self.g
        )

    def compute_acceleration(self, x, v, acceleration):
        """Set the entries acceleration[0] and acceleration[1] of the acceleration vector

        :arg x: angles of bobs wrt vertical (2-dimensional array)
        :arg acceleration: Resulting acceleration vector (2-dimensional array)
        """
        L1 = self.L1
        L2 = self.L2
        mass = self.mass
        g = self.g

        mu = 1 + mass[0] + mass[1]

        acceleration[0] = (
            1
            / (L1 * (mu - np.cos(x[0] - x[1]) ** 2))
            * (
                g * (np.sin(x[1]) * np.cos(x[0] - x[1]) - mu * np.sin(x[0]))
                - (L2 * v[1] ** 2 + L1 * v[0] ** 2 * np.cos(x[0] - x[1]))
                * np.sin(x[0] - x[1])
            )
        )
        acceleration[1] = (
            1
            / (L2 * (mu - np.cos(x[0] - x[1]) ** 2))
            * (
                g * mu * (np.sin(x[0]) * np.cos(x[0] - x[1]) - np.sin(x[1]))
                + (L1 * mu * (v[0] ** 2) + L2 * (v[1] ** 2) * np.cos(x[0] - x[1]))
                * np.sin(x[0] - x[1])
            )
        )

    def set_random_state(self, x, v):
        """Draw position and angular velocity from a normal distribution

        :arg x: Angles with vertical (2-dimensional array)
        :arg v: Angular velocities (2-dimensional array)
        """

        x[0:2] = np.random.normal(0, 0.5 * np.pi, size=(2))  # angles of mass 1 and 2
        v[0:2] = np.random.normal(0, 1, size=(2))  # angular velocities of mass 1 and 2

    def energy(self, x, v):
        """Compute total energy E = 1/2*m*v_0^2 + 1/2*k*x_0^2

        :arg x: Angles with vertical (2-dimensional array)
        :arg v: Angular velocities(2-dimensional array)
        """

        L1 = self.L1
        L2 = self.L2
        g = self.g
        mass = self.mass

        # Potential Energy
        V_pot = g * (
            (mass[0] + mass[1]) * L1 * (1 - np.cos(x[0]))
            + mass[1] * L2 * (1 - np.cos(x[1]))
        )

        # Kinetic Energy
        T_kin = 0.5 * (mass[0] + mass[1]) * L1**2 * v[0] ** 2 + mass[1] * L2 * v[
            1
        ] * (0.5 * L2 * v[1] + L1 * v[0] * np.cos(x[0] - x[1]))

        return V_pot + T_kin


class CoupledPendulums(DynamicalSystem):
    """Two coupled pendulums coupled by a spring and moving in 2d plane.

    The two pendulums are suspended from the ceiling, such that their anchor points are
    a distance d_anchor apart. They are coupled by a sprint with spring constant k_spring,
    such that the force between them is zero if they are hanging down vertically. Both pendulums
    have the same mass and are connected to the ceiling by a massless rod of length L_rod.

    If x_0 = theta_0 and x_1 = theta_1 are the angles of the two pendulums, then the positions of
    the masses are:

    x_0 = (L*sin(theta_0), -L*cos(theta_0))
    x_1 = (d+L*sin(theta_1)), -L*cos(theta_1))

    The potential energy is

    V_pot = mass*g_grav*L_rod*( (1-cos(theta_0)) + (1-cos(theta_1)) )
          + 1/2*k_spring*(|x_0-x_1|-d_anchor)^2

    where g_grav is the gravitational acceleration and the kinetic energy is

    T_kin = 1/2*mass*L_rod^2*( dot(theta_0)^2 + dot(theta_1)^2 )
    """

    def __init__(self, mass, L_rod, d_anchor, k_spring, g_grav=9.81):
        """Create new instance of coupled pendulums class

        :arg L_rod: length of rods
        :arg d_anchor: distance between anchor points
        :arg k_spring: spring constant
        :arg g_grav: gravitational acceleration
        """
        super().__init__(2, mass)
        self.L_rod = L_rod
        self.d_anchor = d_anchor
        self.k_spring = k_spring
        self.g_grav = g_grav
        # C-code snipped for computing the acceleration update
        self.acceleration_header_code = """
        #include "math.h"
        """
        self.acceleration_preamble_code = """
        double phi;
        double sin_x0;
        double sin_x1;
        double cos_x0;
        double cos_x1;
        double C_tmp;
        double sin_x0_x1;
        double z0;
        double z1;
        """
        self.acceleration_update_code = f"""
        sin_x0 = sin(x[0]);
        sin_x1 = sin(x[1]);
        cos_x0 = cos(x[0]);
        cos_x1 = cos(x[1]);
        sin_x0_x1 = sin(x[0]-x[1]);
        z0 = {self.d_anchor} + {self.L_rod} * (sin_x1 - sin_x0);
        z1 = {self.L_rod}* (cos_x1 - cos_x0);
        phi = sqrt( z0*z0 + z1*z1 );
        C_tmp = {self.k_spring} / ({self.L_rod} * {self.mass}) * ({self.d_anchor}/phi - 1.0);        
        a[0] += C_tmp * ( -{self.d_anchor} * cos_x0 + {self.L_rod} * sin_x0_x1);
        a[0] -= {self.g_grav} / {self.L_rod} * sin_x0;
        a[1] += C_tmp * ( +{self.d_anchor} * cos_x1 - {self.L_rod} * sin_x0_x1);
        a[1] -= {self.g_grav} / {self.L_rod} * sin_x1;
        """

    def _phi(self, theta_0, theta_1):
        """Compute distance |x_0-x_1| = phi(theta_0, theta_1)

        given by

        phi(theta_0,theta_1) := |x_0-x_1| = sqrt( (d_anchor + L_rod*(sin(theta_1)-sin(theta_0)))^2
                                                + L_rod^2*(cos(theta_1)-cos(theta_0))^2 )

        :arg theta_0: angle of first bob
        :arg theta_1: angle of second bob
        """
        return np.sqrt(
            (self.d_anchor + self.L_rod * (np.sin(theta_1) - np.sin(theta_0))) ** 2
            + self.L_rod**2 * (np.cos(theta_1) - np.cos(theta_0)) ** 2
        )

    def compute_acceleration(self, x, v, acceleration):
        """Set the entries acceleration[0] and acceleration[1] of the acceleration vector

        The acceleration is the angular acceleration, i.e. the force scaled by the momentum of
        inertia given by I = mass*L_rod^2

        With theta_0 = x[0], theta_1, x[1], dot(theta_0) = v[0], dot(theta_1) = v[1],
        the accelerations are gives:

        a_0 = -1/I*dV_pot/dtheta_0
            = C * ( -d_anchor*cos(theta_0) + L_rod*sin(theta_0-theta_1) )
              - g_grav/L_rod*sin(theta_0)

        a_1 = -1/I*dV_pot/dtheta_1
            = C * ( d_anchor*cos(theta_0) - L_rod*sin(theta_0-theta_1) )
              - g_grav/L_rod*sin(theta_1)

        where

        C = k_spring/(L_rod*mass)*(d_anchor-phi(theta_0,theta_1)) / phi(theta_0,theta_1)

        :arg x: angles of bobs wrt vertical (2-dimensional array)
        :arg acceleration: Resulting acceleration vector (2-dimensional array)
        """
        phi = self._phi(x[0], x[1])
        C_tmp = self.k_spring / (self.L_rod * self.mass) * (self.d_anchor - phi) / phi
        acceleration[0] = C_tmp * (
            -self.d_anchor * np.cos(x[0]) + self.L_rod * np.sin(x[0] - x[1])
        ) - self.g_grav / self.L_rod * np.sin(x[0])
        acceleration[1] = C_tmp * (
            self.d_anchor * np.cos(x[1]) - self.L_rod * np.sin(x[0] - x[1])
        ) - self.g_grav / self.L_rod * np.sin(x[1])

    def set_random_state(self, x, v):
        """Draw angles and angular velocities.

        We assume that angles theta_0 and theta_1 stay in the range [-pi/4,+pi/4], and
        that the energy never exceeds the maximum value

        E_{max} = k_spring*L_rod^2 + mass*g_grav*L_rod*(2-sqrt(2))

        :arg x: Angles with vertical (2-dimensional array)
        :arg v: Angular velocities (2-dimensional array)
        """

        # Draw angle
        x[0:2] = np.random.uniform(low=-0.25 * np.pi, high=+0.25 * np.pi, size=(2))
        R_theta = np.sqrt(
            self.k_spring
            / (self.mass * self.L_rod**2)
            * (2.0 * self.L_rod**2 - (self._phi(x[0], x[1]) - self.d_anchor) ** 2)
            + 2.0
            * self.g_grav
            / self.L_rod
            * (np.cos(x[0]) + np.cos(x[1]) - np.sqrt(2))
        )
        v[:] = R_theta
        while v[0] ** 2 + v[1] ** 2 > R_theta**2:
            v[0:2] = np.random.uniform(low=-R_theta, high=R_theta, size=(2))

    def energy(self, x, v):
        """Compute total energy E = V_pot + T_kin

        :arg x: Angles with vertical (2-dimensional array)
        :arg v: Angular velocities(2-dimensional array)
        """
        V_pot = 0.5 * self.k_spring * (
            self._phi(x[0], x[1]) - self.d_anchor
        ) ** 2 + self.mass * self.g_grav * self.L_rod * (
            2 - np.cos(x[0]) - np.cos(x[1])
        )
        T_kin = 0.5 * self.mass * self.L_rod**2 * (v[0] ** 2 + v[1] ** 2)
        return V_pot + T_kin


class LennartJonesSystem(DynamicalSystem):
    """Set of particles interacting via the truncated Lennart-Jones
    potential V(r) defined by

    V(r) = V_{LJ}(r) - V_{LJ}(r_c)

    where V_{LJ}(r) = 4*epsilon*((1/r)^{12}-(1/r)^6)

    All length scales are measured in units of a length scale sigma.
    The parameters of the potential are:

    * energy scale epsilon (V_{LJ}(1) = -epsilon)
    * cutoff r_c

    It is assumed that there are npart particles which move in a box of
    size boxsize x boxsize with periodic boundary conditions.
    The positions and velocities are ordered such that, for example, the
    position vector is

    x = [x_0,y_0,x_1,y_1,x_2,y_2,...]

    where (x_j,y_j) is the position of particle j.

    The typical kinetic energy scale, which is used to initialse the particle velocities
    at random, can also be set explicitly.
    """

    def __init__(
        self,
        mass,
        npart,
        boxsize,
        epsilon_pot=1.0,
        epsilon_kin=1.0,
        rcutoff=3.0,
        fast_acceleration=True,
    ):
        """Construct new instance of Lennart-Jones system

        :arg mass: Particle mass
        :arg npart: Number of particles
        :arg boxsize: Size of box
        :arg epsilon_kin: Scale of potential energy
        :arg epsilon_pot: Scale of kinetic enegry
        :arg rcutoff: Cutoff distance for potential
        :arg fast_acceleration: Evaluate acceleration using compiled C code
        """
        super().__init__(2 * npart, mass)
        # Set parameters of dynamical system
        self.npart = npart
        self.boxsize = boxsize
        # Set potential and kinetic energy parameters
        self.epsilon_pot = epsilon_pot
        self.epsilon_kin = epsilon_kin
        self.rcutoff = rcutoff
        # Ensure that particle can not interact with its own periodic copies
        assert self.rcutoff < self.boxsize
        # Shift in potential energy to ensure that V(r_c) = 0
        self.Vshift = (
            4.0
            * self.epsilon_pot
            * (1.0 / self.rcutoff**12 - 1.0 / self.rcutoff**6)
        )
        self.fast_acceleration = fast_acceleration
        if self.fast_acceleration:
            c_sourcecode = string.Template(
                """
            void calculate_lj_acceleration(double* x, double* acceleration) {
                const int npart = $NPART;
                const double boxsize = $BOXSIZE;
                const double rcutoff2 = $RCUTOFF2;
                const double fscal = 24.*$EPSILON_POT/$MASS;
                for (int j=0;j<npart;++j) {
                    acceleration[2*j] = 0.0;
                    acceleration[2*j+1] = 0.0;
                    for (int k=0;k<npart;++k) {
                        if (j==k) continue;
                        for (int xoff=0;xoff<=1;++xoff) {
                            for (int yoff=0;yoff<=1;++yoff) {
                                double dx = x[2*k]-x[2*j] + boxsize*xoff;
                                double dy = x[2*k+1]-x[2*j+1] + boxsize*yoff;
                                double nrm2 = dx*dx + dy*dy;
                                if  (nrm2 <= rcutoff2) {
                                    double invnrm2 = 1./nrm2;
                                    double invnrm4 = invnrm2*invnrm2;
                                    double invnrm6 = invnrm4*invnrm2;
                                    double invnrm8 = invnrm4*invnrm4;
                                    double Fabs = fscal*invnrm8*(2.*invnrm6-1.);
                                    acceleration[2*j] -= Fabs*dx;
                                    acceleration[2*j+1] -= Fabs*dy;
                                }
                            }
                        }
                    }
                }
            }

            double calculate_lj_potential_energy(double* x) {
                const int npart = $NPART;
                const double boxsize = $BOXSIZE;
                const double rcutoff2 = $RCUTOFF2;
                const double Vscal = 4.*$EPSILON_POT;
                const double Vshift = $VSHIFT;
                double Vpot = 0.0;
                for (int j=0;j<npart;++j) {
                    for (int k=0;k<j;++k) {
                        for (int xoff=0;xoff<=1;++xoff) {
                            for (int yoff=0;yoff<=1;++yoff) {
                                double dx = x[2*k]-x[2*j] + boxsize*xoff;
                                double dy = x[2*k+1]-x[2*j+1] + boxsize*yoff;
                                double nrm2 = dx*dx + dy*dy;
                                if (nrm2 <= rcutoff2) {
                                    double invnrm2 = 1./nrm2;
                                    double invnrm6 = invnrm2*invnrm2*invnrm2;
                                    Vpot += Vscal*invnrm6*(invnrm6-1.);
                                    Vpot -= Vshift;
                                }
                            }
                        }
                    }
                }
                return Vpot;
            }
            """
            ).substitute(
                NPART=self.npart,
                BOXSIZE=self.boxsize,
                RCUTOFF2=self.rcutoff**2,
                EPSILON_POT=self.epsilon_pot,
                MASS=self.mass,
                VSHIFT=self.Vshift,
            )
            with open("calculate_lj_acceleration.c", "w", encoding="utf8") as f:
                print(c_sourcecode, file=f)
            # Compile source code (might have to adapt for different compiler)
            subprocess.run(
                [
                    "gcc",
                    "-fPIC",
                    "-shared",
                    "-o",
                    "calculate_lj_acceleration.so",
                    "calculate_lj_acceleration.c",
                ],
                check=True,
            )
            so_file = "./calculate_lj_acceleration.so"
            self.calculate_lj_acceleration = ctypes.CDLL(
                so_file
            ).calculate_lj_acceleration
            self.calculate_lj_acceleration.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ]
            self.calculate_lj_potential_energy = ctypes.CDLL(
                so_file
            ).calculate_lj_potential_energy
            self.calculate_lj_potential_energy.restype = ctypes.c_double
            self.calculate_lj_potential_energy.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
            ]

    def compute_acceleration(self, x, v, acceleration):
        """Set the entries acceleration[:] of the acceleration vector

        :arg x: Particle position x (2*npart-dimensional array)
        :arg v: Particle velocities x (d-dimensional array)
        :arg acceleration: Resulting acceleration vector (2*npart-dimensional array)
        """
        if self.fast_acceleration:
            self.calculate_lj_acceleration(x, acceleration)
        else:
            acceleration[:] = 0.0
            for j in range(self.npart):
                for k in range(self.npart):
                    if j == k:
                        continue
                    for xboxoffset in (-1, 0, +1):
                        for yboxoffset in (-1, 0, +1):
                            dx = x[2 * k : 2 * k + 2] - x[2 * j : 2 * j + 2]
                            dx[0] += self.boxsize * xboxoffset
                            dx[1] += self.boxsize * yboxoffset
                            nrm2 = dx[0] ** 2 + dx[1] ** 2
                            if nrm2 <= self.rcutoff**2:
                                invnrm2 = 1.0 / nrm2
                                Fabs = (
                                    24.0
                                    * self.epsilon_pot
                                    * invnrm2**4
                                    * (2.0 * invnrm2**3 - 1.0)
                                )
                                acceleration[2 * j : 2 * j + 2] -= (
                                    Fabs / self.mass * dx[:]
                                )

    def set_random_state(self, x, v):
        """Draw position and velocity randomly

        The velocities are drawn from a normal distribution with mean zero and
        variance chosen such that <1/2*m*v^2> = E_kin. The particles are
        distributed randomly inside the box, such that the distance between any
        two particles (including their periodic copies) is at least the
        distance at which the potential is minimal.

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        """

        # Draw positions such that no two particles (and their periodic copies)
        # are closer than 1 (~ the potential minimum)
        accepted = False
        while not accepted:
            positions = np.random.uniform(
                low=0, high=self.boxsize, size=(self.npart, 2)
            )
            for offsetx in (-1, 0, +1):
                for offsety in (-1, 0, +1):
                    if (offsetx == 0) and (offsety == 0):
                        continue
                    offset_vector = np.array(
                        [self.boxsize * offsetx, self.boxsize * offsety]
                    )
                    positions = np.concatenate(
                        (positions, positions[: self.npart, :] + offset_vector)
                    )
            accepted = True
            for j in range(9 * self.npart):
                for k in range(j):
                    diff = positions[j, :] - positions[k, :]
                    distance = np.linalg.norm(diff)
                    accepted = accepted and (distance > 2.0 ** (1.0 / 6.0))
        x[:] = positions[: self.npart, :].flatten()
        # Draw velocities such that <1/2*m*v^2> ~ \epsilon_kin
        sigma_v = np.sqrt(2.0 * self.epsilon_kin / self.mass)
        v[:] = np.random.normal(0, sigma_v, size=self.dim)

    def visualise_configuration(self, x, v, filename=None):
        """Visualise the configuration (including any periodic copies)

        :arg x: Positions
        :arg v: Velocities
        """
        plt.clf()
        ax = plt.gca()
        ax.set_aspect(1.0)
        ax.set_xlim(0, self.boxsize)
        ax.set_ylim(0, self.boxsize)
        X, Y = x.reshape((self.npart, 2)).transpose()
        Vx, Vy = v.reshape((self.npart, 2)).transpose()
        plt.plot(X, Y, markersize=4, marker="o", linewidth=0, color="red")
        for j in range(self.npart):
            plt.arrow(X[j], Y[j], Vx[j], Vy[j])
        for offsetx in (-1, 0, +1):
            for offsety in (-1, 0, +1):
                offset_vector = np.array(
                    [self.boxsize * offsetx, self.boxsize * offsety]
                )
                for j in range(self.npart):
                    x_shifted = x[2 * j : 2 * j + 2] + offset_vector
                    circle = plt.Circle(x_shifted, 1.0, color="r", alpha=0.2)
                    ax.add_patch(circle)
        if filename:
            plt.savefig(filename, bbox_inches="tight")

    def energy(self, x, v):
        """Compute total energy

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        """
        Vkin = 0.5 * self.mass * np.dot(v, v)
        if self.fast_acceleration:
            Vpot = self.calculate_lj_potential_energy(x)
        else:
            Vpot = 0.0
            for j in range(self.npart):
                for k in range(j):
                    for xboxoffset in (-1, 0, +1):
                        for yboxoffset in (-1, 0, +1):
                            dx = x[2 * k : 2 * k + 2] - x[2 * j : 2 * j + 2]
                            dx[0] += self.boxsize * xboxoffset
                            dx[1] += self.boxsize * yboxoffset
                            nrm2 = dx[0] ** 2 + dx[1] ** 2
                            if nrm2 <= self.rcutoff**2:
                                invnrm6 = 1.0 / nrm2**3
                                Vpot += (
                                    4.0 * self.epsilon_pot * invnrm6 * (invnrm6 - 1.0)
                                )
                                Vpot -= self.Vshift
        return Vkin + Vpot

    def apply_constraints(self, x):
        """Apply constraints, such as period boundary conditions

        :arg x: Positions (d-dimensional array)
        """
        x[:] -= (x[:] // self.boxsize) * self.boxsize
