import numpy as np
import string
import subprocess
import ctypes
import hashlib

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
        self.dynamical_system.compute_scaled_force(self.x,self.v,self.force)

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
            self.dynamical_system.compute_scaled_force(self.x,self.v,self.force)

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
        # Check whether dynamical system has a C-code snippet for updating the acceleration
        self.fast_code = hasattr(self.dynamical_system,'acceleration_update_code')
        # If this is the case, auto-generate fast C code for the Velocity Verlet update
        if self.fast_code:
            if self.dynamical_system.acceleration_preamble_code:
                preamble = self.dynamical_system.acceleration_preamble_code
            else:
                preamble = ''
            if self.dynamical_system.acceleration_header_code:
                header = self.dynamical_system.acceleration_header_code
            else:
                header = ''
            c_sourcecode = string.Template('''
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
            ''').substitute(DIM=self.dynamical_system.dim,
                            DT=self.dt,
                            ACCELERATION_UPDATE_CODE=self.dynamical_system.acceleration_update_code,
                            ACCELERATION_HEADER_CODE=header,
                            ACCELERATION_PREAMBLE_CODE=preamble)
            sha = hashlib.md5()
            sha.update(c_sourcecode.encode())
            filestem = './velocity_verlet_'+sha.hexdigest()
            so_file = filestem+'.so'
            source_file = filestem+'.c'
            with open(source_file,'w') as f:
                print (c_sourcecode,file=f)
            # Compile source code (might have to adapt for different compiler)
            subprocess.run(['gcc',
                            '-fPIC','-shared','-o',
                            so_file,
                            source_file])
            self.c_velocity_verlet = ctypes.CDLL(so_file).velocity_verlet
            self.c_velocity_verlet.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double,
                                               flags="C_CONTIGUOUS"),
                                               np.ctypeslib.ndpointer(ctypes.c_double,
                                               flags="C_CONTIGUOUS"),
                                               np.ctypeslib.c_intp]

    def integrate(self,n_steps):
        '''Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        '''
        if self.fast_code:
            self.c_velocity_verlet(self.x,self.v,n_steps)
        else:
            for k in range(n_steps):
                self.x[:] += self.dt*self.v[:] + 0.5*self.dt**2*self.force[:]
                self.dynamical_system.apply_constraints(self.x)
                self.v[:] += 0.5*self.dt*self.force[:]
                self.dynamical_system.compute_scaled_force(self.x,self.v,self.force)
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
