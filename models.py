'''Neural network models'''
import tensorflow as tf
from tensorflow import keras

class VerletModel(keras.Model):
    '''Single step of a Symplectic Stoermer Verlet integrator update for a 
    separable system with Hamiltonian H(q,p) = T(p) + V(q)
    
    The model maps the current state (q_n,p_n) to next state (q_{n+1},p_{n+1}) 
    using the update
    
    p_{n+1/2} = p_n - dt/2*dV/dq(q_n)
    q_{n+1} = q_n + dt*dT/dp(p_{n+1/2})
    p_{n+1} = p_{n+1/2} - dt/2*dV/dq(q_{n+1})
    
    Both the kinetic energy T(p) and the potential energy V(q) are represented 
    by neural networks. The position q_n and momentum p_n are d-dimensional vectors.
    
    :arg dim: dimension d of the Hamiltonian system 
    :arg dt: timestep size
    :arg V_pot_layers: layers encoding the neural network for potential energy V(q)
    :arg T_kin_layers: layers encoding the neural network for kinetic energy T(p)
    '''
    
    def __init__(self,dim,dt,V_pot_layers,T_kin_layers):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.V_pot_layers = V_pot_layers
        self.T_kin_layers = T_kin_layers
        self.V_pot_final_layer = keras.layers.Dense(self.dim//2,use_bias=False)
        self.T_kin_final_layer = keras.layers.Dense(self.dim//2,use_bias=False)
    
    #@tf.function
    def V_pot(self,q):
        '''Evaluate potential energy network V(q)
        
        :arg q: position q  which to evaluate the potential
        '''
        x = q
        for layer in self.V_pot_layers:
            x = layer(x)
        x = self.V_pot_final_layer(x)
        return x
        
        
    #@tf.function
    def T_kin(self,p):
        '''Evaluate kinetic energy network T(p)
        
        :arg p: momentum p at which to evaluate the kinetic energy
        '''
        x = p
        for layer in self.T_kin_layers:
            x = layer(x)
        x = self.T_kin_final_layer(x)
        return x
        

    @tf.function
    def verlet_step(self,q_n,p_n):
        '''Carry out a single Stoermer-Verlet step
        
        This function maps (q_n,p_n) to (q_{n+1},p_{n+1}) using a single Stoermer
        Verlet step
        
        :arg q_n: current position q_n
        :arg p_n: current momentum p_n
        '''
        # p_{n+1/2} = p_n - dt/2*dV/dq(q_n)
        dV_dq = tf.gradients(self.V_pot(q_n),q_n)[0]
        p_n = p_n - 0.5*self.dt*dV_dq

        # q_{n+1} = q_n + dt*dT/dq(p_{n+1/2})
        dT_dp = tf.gradients(self.T_kin(p_n),p_n)[0]
        q_n = q_n + self.dt*dT_dp

        # p_{n+1} = p_{n+1/2} - dt/2*dV/dq(q_{n+1})
        dV_dq = tf.gradients(self.V_pot(q_n),q_n)[0]
        p_n = p_n - 0.5*self.dt*dV_dq
        
        return q_n, p_n

    def call(self, inputs):
        '''Evaluate model
        
        Split the inputs = (q_n,p_n) into position and momentum and 
        return the state (q_{n+1},p_{n+1}) at the next timestep.
        
        Note that the expected tensor shape is B x 1 x 2d to be compatible with
        the non-symplectic update 
        
        :arg inputs: state (q_n,p_n) as a B x 1 x 2d tensor
        '''
        
        input_shape = tf.shape(inputs)
        # Extract q_n and p_n from input
        qp_old = tf.unstack(tf.reshape(inputs, (input_shape[0],input_shape[2],)),axis=-1)
        q_old = tf.stack(qp_old[:self.dim//2],axis=-1)
        p_old = tf.stack(qp_old[self.dim//2:],axis=-1)
        q_new, p_new = self.verlet_step(q_old,p_old)        
        # Combine result of Verlet step into tensor of correct size
        outputs = tf.concat([q_new,p_new],axis=-1)
        return outputs