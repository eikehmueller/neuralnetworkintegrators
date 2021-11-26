import numpy as np

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

    def compute_scaled_force(self,x,force):
        '''Store the forces scaled by inverse mass in the vector
        such that force[j] = F_j(x)/m_j

        :arg x: Particle positions x (d-dimensional array)
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

    def compute_scaled_force(self,x,force):
        '''Set the entry force[0] of the force vector
        to -k/m_0*x_0

        :arg x: Particle position x (1-dimensional array)
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
