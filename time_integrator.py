import numpy as np

class TimeIntegrator(object):
    def __init__(self,dynamical_system,dt):
        '''Abstract base class for a single step traditional time integrator

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        '''
        self.dynamical_system = dynamical_system
        self.dt = dt
        self.x = np.zeros(dynamical_system.dim)
        self.v = np.zeros(dynamical_system.dim)
        self.force = np.zeros(dynamical_system.dim)
        self.label = None

    def set_state(self,x,v):
        '''Set the current state of the integrator to a specified
        position and velocity.

        :arg x: New position vector
        :arg v: New velocity vector
        '''
        self.x[:] = x[:]
        self.v[:] = v[:]
        self.dynamical_system.compute_scaled_force(self.x,self.force)

    def integrate(self,n_steps):
        '''Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        '''
        pass

    def energy(self):
        '''Return the energy of the underlying dynamical system for
        the current position and velocity'''
        return self.dynamical_system.energy(self.x,self.v)


class ForwardEulerIntegrator(TimeIntegrator):
    def __init__(self,dynamical_system,dt):
        '''Forward Euler integrator given by

        x_j^{(t+dt)} = x_j^{(t)} + dt*v_j
        v_j^{(t+dt)} = v_j^{(t)} + dt*F_j(x^{(t)})/m_j

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        '''
        super().__init__(dynamical_system,dt)
        self.label = 'ForwardEuler'

    def integrate(self,n_steps):
        '''Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        '''
        for k in range(n_steps):
            self.x[:] += self.dt*self.v[:]
            self.dynamical_system.apply_constraints(self.x)
            self.v[:] += self.dt*self.force[:]
            # Compute force at next timestep
            self.dynamical_system.compute_scaled_force(self.x,self.force)

class VerletIntegrator(TimeIntegrator):
    def __init__(self,dynamical_system,dt):
        '''Verlet integrator given by

        x_j^{(t+dt)} = x_j^{(t)} + dt*v_j + dt^2/2*F_j(x^{(t)})/m_j
        v_j^{(t+dt)} = v_j^{(t)} + dt^2/2*(F_j(x^{(t)})/m_j+F_j(x^{(t+dt)})/m_j)

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        '''
        super().__init__(dynamical_system,dt)
        self.label = 'Verlet'

    def integrate(self,n_steps):
        '''Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        '''
        for k in range(n_steps):
            self.x[:] += self.dt*self.v[:] + 0.5*self.dt**2*self.force[:]
            self.dynamical_system.apply_constraints(self.x)
            self.v[:] += 0.5*self.dt*self.force[:]
            self.dynamical_system.compute_scaled_force(self.x,self.force)
            self.v[:] += 0.5*self.dt*self.force[:]

class ExactIntegrator(TimeIntegrator):
    def __init__(self,dynamical_system,dt):
        '''Exact integrator
        
        Integrate the equations of motion exactly, if the dynamical system supports this.

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        '''
        super().__init__(dynamical_system,dt)
        self.label = 'Exact'

    def integrate(self,n_steps):
        '''Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        '''
        self.x[:], self.v[:] = self.dynamical_system.forward_map(self.x[:],self.v[:],n_steps*self.dt)