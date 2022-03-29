from matplotlib import pyplot as plt
import ctypes
import subprocess
import numpy as np
import string

class DynamicalSystem(object):
    def __init__(self,dim,mass):
        '''Abstract base class for a dynamical system

        Models a d-dimensional system of the following form:

          dx_j/dt = v_j
          dv_j/dt = F_j(x)/m_j

        where j = 0,1,2,...,d-1

        :arg dim: Spatial dimension of dynamical system
        :arg mass: Mass of the system (can be a scalar in 1d)
        '''
        self.dim = dim
        self.mass = mass

    def compute_scaled_force(self,x,v,force):
        '''Store the forces scaled by inverse mass in the vector
        such that force[j] = F_j(x)/m_j

        :arg x: Particle positions x (d-dimensional array)
        :arg v: Particle velocities x (d-dimensional array)
        :arg force: Resulting force vector (d-dimensional array)
        '''
        pass

    def set_random_state(self,x,v):
        '''Set the position x and v to random values. This will be used
        during the training stage to pick a suitable set of initial values
        for the problem at hand

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        '''
        pass


    def energy(self,x,v):
        '''Return the total energy for given positions and velocities

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        '''
        pass

    def apply_constraints(self,x):
        '''Apply constraints, such as period boundary conditions

        :arg x: Positions (d-dimensional array)
        '''
        pass

    def forward_map(self,x0,v0,t):
        '''Exact forward map

        Compute position x(t) and velocity v(t), given initial position x(0) and velocity v(0).
        This will only be implemented if the specific dynamical system has an analytical solution

        :arg x0: initial position x(0)
        :arg v0: initial velocity v(0)
        :arg t: final time
        '''
        raise NotImplementedError("Dynamical system has no exact solution.")

class HarmonicOscillator(DynamicalSystem):
    def __init__(self,mass,k_spring):
        '''One-dimensional harmonic oscillator described by the equations
        of motion

        dx_0/dt = v_0, dv_0/dt = -k/m_0*x_0

        :arg mass: Particle mass
        :arg k_spring: Spring constant k
        '''
        super().__init__(1,mass)
        self.k_spring = k_spring
        # C-code snipped for computing the acceleration update
        self.acceleration_update_code = string.Template('''
        a[0] += -($KSPRING/$MASS)*x[0];
        ''').substitute(KSPRING=self.k_spring,MASS=self.mass)

    def compute_scaled_force(self,x,v,force):
        '''Set the entry force[0] of the force vector
        to -k/m_0*x_0

        :arg x: Particle position x (1-dimensional array)
        :arg v: Particle velocities x (d-dimensional array)
        :arg force: Resulting force vector (1-dimensional array)
        '''
        force[0] = -self.k_spring/self.mass*x[0]

    def set_random_state(self,x,v):
        '''Draw position and velocity from a normal distribution
        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        '''
        x[0] = np.random.normal(0,1)
        v[0] = np.random.normal(0,1)

    def energy(self,x,v):
        '''Compute total energy E = 1/2*m*v_0^2 + 1/2*k*x_0^2

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        '''
        return 0.5*self.mass*v[0]**2 + 0.5*self.k_spring*x[0]**2

    def forward_map(self,x0,v0,t):
        '''Exact forward map

        Compute position x(t) and velocity v(t), given initial position x(0) and velocity v(0).

        For this use:

        x(t) = x(0)*cos(omega*t) + omega*v(0)*sin(omega*t)
        v(t) = -x(0)/omega*sin(omega*t) + v(0)*cos(omega*t)

        with omegae = sqrt(k/m)

        :arg x0: initial position x(0)
        :arg v0: initial velocity v(0)
        :arg t: final time
        '''
        omega = np.sqrt(self.k_spring/self.mass)
        cos_omegat = np.cos(omega*t)
        sin_omegat = np.sin(omega*t)
        x = np.array(x0[0]*cos_omegat + v0[0]/omega*sin_omegat)
        v = np.array(-x0[0]*omega*sin_omegat + v0[0]*cos_omegat)
        return x, v


class LennartJonesSystem(DynamicalSystem):
    def __init__(self,mass,npart,boxsize,
                 epsilon_pot=1.0,
                 epsilon_kin=1.0,
                 rcutoff=3.0,
                 fast_force=True):
        '''Set of particles interacting via the truncated Lennart-Jones
        potential V(r) defined by

        V(r) = V_{LJ}(r) - V_{LJ}(r_c)

        where V_{LJ}(r) = 4*\epsilon*((1/r)^{12}-(1/r)^6)

        All length scales are measured in units of a length scale \sigma.
        The parameters of the potential are:

        * energy scale \epsilon (V_{LJ}(1) = -\epsilon)
        * cutoff r_c

        It is assumed that there are npart particles which move in a box of
        size boxsize x boxsize with periodic boundary conditions.
        The positions and velocities are ordered such that, for example, the
        position vector is

        x = [x_0,y_0,x_1,y_1,x_2,y_2,...]

        where (x_j,y_j) is the position of particle j.

        The typical kinetic energy scale, which is used to initialse the particle velocities at random, can also be set.

        :arg mass: Particle mass
        :arg npart: Number of particles
        :arg boxsize: Size of box
        :arg epsilon_kin: Scale of potential energy
        :arg epsilon_pot: Scale of kinetic enegry
        :arg rcutoff: Cutoff distance for potential
        :arg fast_force: Evaluate force using compiled C code
        '''
        super().__init__(2*npart,mass)
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
        self.Vshift = 4.*self.epsilon_pot*(1./self.rcutoff**12-1./self.rcutoff**6)
        self.fast_force=fast_force
        if (self.fast_force):
            c_sourcecode = string.Template('''
            void calculate_lj_force(double* x, double* force) {
                const int npart = $NPART;
                const double boxsize = $BOXSIZE;
                const double rcutoff2 = $RCUTOFF2;
                const double fscal = 24.*$EPSILON_POT/$MASS;
                for (int j=0;j<npart;++j) {
                    force[2*j] = 0.0;
                    force[2*j+1] = 0.0;
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
                                    force[2*j] -= Fabs*dx;
                                    force[2*j+1] -= Fabs*dy;
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
            ''').substitute(NPART=self.npart,
                            BOXSIZE=self.boxsize,
                            RCUTOFF2=self.rcutoff**2,
                            EPSILON_POT=self.epsilon_pot,
                            MASS=self.mass,
                            VSHIFT=self.Vshift)
            with open('calculate_lj_force.c','w') as f:
                print (c_sourcecode,file=f)
            # Compile source code (might have to adapt for different compiler)
            subprocess.run(['gcc',
                            '-fPIC','-shared','-o',
                            'calculate_lj_force.so',
                            'calculate_lj_force.c'])
            so_file = './calculate_lj_force.so'
            self.calculate_lj_force = ctypes.CDLL(so_file).calculate_lj_force
            self.calculate_lj_force.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double,
                                    flags="C_CONTIGUOUS"),
             np.ctypeslib.ndpointer(ctypes.c_double,
                                    flags="C_CONTIGUOUS")]
            self.calculate_lj_potential_energy = ctypes.CDLL(so_file).calculate_lj_potential_energy
            self.calculate_lj_potential_energy.restype = ctypes.c_double
            self.calculate_lj_potential_energy.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double,
                                    flags="C_CONTIGUOUS")]


    def compute_scaled_force(self,x,force):
        '''Set the entries force[:] of the force vector

        :arg x: Particle position x (2*npart-dimensional array)
        :arg force: Resulting force vector (2*npart-dimensional array)
        '''
        if (self.fast_force):
            self.calculate_lj_force(x,force)
        else:
            force[:] = 0.0
            for j in range(self.npart):
                for k in range(self.npart):
                    if j==k:
                        continue
                    for xboxoffset in (-1,0,+1):
                        for yboxoffset in (-1,0,+1):
                            dx = x[2*k:2*k+2]-x[2*j:2*j+2]
                            dx[0] += self.boxsize*xboxoffset
                            dx[1] += self.boxsize*yboxoffset
                            nrm2 = dx[0]**2 + dx[1]**2
                            if  nrm2 <= self.rcutoff**2:
                                invnrm2 = 1./nrm2
                                Fabs = 24.*self.epsilon_pot*invnrm2**4*(2.*invnrm2**3 - 1.0)
                                force[2*j:2*j+2] -= Fabs/self.mass*dx[:]

    def set_random_state(self,x,v):
        '''Draw position and velocity randomly

        The velocities are drawn from a normal distribution with mean zero and
        variance chosen such that <1/2*m*v^2> = E_kin. The particles are
        distributed randomly inside the box, such that the distance between any
        two particles (including their periodic copies) is at least the
        distance at which the potential is minimal.

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        '''

        # Draw positions such that no two particles (and their periodic copies)
        # are closer than 1 (~ the potential minimum)
        accepted = False
        while not accepted:
            positions = np.random.uniform(low=0,
                                          high=self.boxsize,
                                          size=(self.npart,2))
            for offsetx in (-1,0,+1):
                for offsety in (-1,0,+1):
                    if (offsetx==0) and (offsety==0):
                        continue
                    offset_vector = np.array([self.boxsize*offsetx,
                                              self.boxsize*offsety])
                    positions = np.concatenate((positions,
                                                positions[:self.npart,:]+offset_vector))
            accepted = True
            for j in range(9*self.npart):
                for k in range(j):
                    diff = positions[j,:]-positions[k,:]
                    distance = np.linalg.norm(diff)
                    accepted = accepted and (distance > 2.**(1./6.))
        x[:] = positions[:self.npart,:].flatten()
        # Draw velocities such that <1/2*m*v^2> ~ \epsilon_kin
        sigma_v = np.sqrt(2.*self.epsilon_kin/self.mass)
        v[:] = np.random.normal(0,sigma_v,size=self.dim)

    def visualise_configuration(self,x,v,filename=None):
        '''Visualise the configuration (including any periodic copies)

        :arg x: Positions
        :arg v: Velocities
        '''
        plt.clf()
        ax = plt.gca()
        ax.set_aspect(1.0)
        ax.set_xlim(0,self.boxsize)
        ax.set_ylim(0,self.boxsize)
        X,Y = x.reshape((self.npart,2)).transpose()
        Vx,Vy = v.reshape((self.npart,2)).transpose()
        plt.plot(X,Y,markersize=4,marker='o',linewidth=0,color='red')
        for j in range(self.npart):
            plt.arrow(X[j], Y[j], Vx[j], Vy[j])
        for offsetx in (-1,0,+1):
            for offsety in (-1,0,+1):
                offset_vector = np.array([self.boxsize*offsetx,
                                          self.boxsize*offsety])
                for j in range(self.npart):
                    x_shifted = x[2*j:2*j+2] + offset_vector
                    circle = plt.Circle(x_shifted, 1.0, color='r',alpha=0.2)
                    ax.add_patch(circle)
        if not (filename is None):
            plt.savefig(filename,bbox_inches='tight')

    def energy(self,x,v):
        '''Compute total energy

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        '''
        Vkin = 0.5*self.mass*np.dot(v,v)
        if (self.fast_force):
            Vpot = self.calculate_lj_potential_energy(x)
        else:
            Vpot = 0.0
            for j in range(self.npart):
                for k in range(j):
                    for xboxoffset in (-1,0,+1):
                        for yboxoffset in (-1,0,+1):
                            dx = x[2*k:2*k+2]-x[2*j:2*j+2]
                            dx[0] += self.boxsize*xboxoffset
                            dx[1] += self.boxsize*yboxoffset
                            nrm2 = dx[0]**2 + dx[1]**2
                            if  nrm2 <= self.rcutoff**2:
                                invnrm6 = 1./nrm2**3
                                Vpot += 4.*self.epsilon_pot*invnrm6*(invnrm6-1.)
                                Vpot -= self.Vshift
        return Vkin + Vpot

    def apply_constraints(self,x):
        '''Apply constraints, such as period boundary conditions

        :arg x: Positions (d-dimensional array)
        '''
        x[:] -= (x[:]//self.boxsize)*self.boxsize
