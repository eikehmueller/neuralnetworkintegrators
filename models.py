"""Neural network models"""
import numpy as np
import tensorflow as tf
from tensorflow import keras


class SymplecticModel(keras.Model):
    """Base class for symplectic models"""

    def __init__(self, dim, dt):
        """Construct new instance

        :arg dim: dimension d of the Hamiltonian system
        :arg dt: timestep size
        """
        super().__init__()
        self.dim = dim
        self.dt = dt

    def call(self, inputs):
        """Evaluate model

        Split the inputs = (q_n,p_n) into position and momentum and
        return the state (q_{n+1},p_{n+1}) at the next timestep.

        Note that the expected tensor shape is B x 1 x 2d to be compatible with
        the non-symplectic update.

        :arg inputs: state (q_n,p_n) as a B x 1 x 2d tensor
        """
        input_shape = tf.shape(inputs)
        # Extract q_n and p_n from input
        qp_old = tf.unstack(
            tf.reshape(
                inputs,
                (
                    input_shape[0],
                    input_shape[2],
                ),
            ),
            axis=-1,
        )
        q_old = tf.stack(qp_old[: self.dim // 2], axis=-1)
        p_old = tf.stack(qp_old[self.dim // 2 :], axis=-1)
        q_new, p_new = self.step(q_old, p_old)
        # Combine result of Verlet step into tensor of correct size
        outputs = tf.concat([q_new, p_new], -1)
        return outputs


class VerletModel(SymplecticModel):
    """Single step of a Symplectic Stoermer Verlet integrator update for a
    separable system with Hamiltonian H(q,p) = T(p) + V(q)

    The model maps the current state (q_n,p_n) to next state (q_{n+1},p_{n+1})
    using the update

    p_{n+1/2} = p_n - dt/2*dV/dq(q_n)
    q_{n+1} = q_n + dt*dT/dp(p_{n+1/2})
    p_{n+1} = p_{n+1/2} - dt/2*dV/dq(q_{n+1})

    Both the kinetic energy T(p) and the potential energy V(q) are represented
    by neural networks. The position q_n and momentum p_n are d-dimensional vectors.
    """

    def __init__(self, dim, dt, V_pot_layers, T_kin_layers):
        """Construct new instance

        :arg dim: dimension d of the Hamiltonian system
        :arg dt: timestep size
        :arg V_pot_layers: layers encoding the neural network for potential energy V(q)
        :arg T_kin_layers: layers encoding the neural network for kinetic energy T(p)
        """
        super().__init__(dim, dt)
        self.V_pot_layers = V_pot_layers
        self.T_kin_layers = T_kin_layers
        self.V_pot_final_layer = keras.layers.Dense(1, use_bias=False)
        self.T_kin_final_layer = keras.layers.Dense(1, use_bias=False)

    def set_weights(self, V_pot_layer_weights, T_kin_layer_weights):
        """Set weights of layers in energy networks

        :arg V_pot_layer_weights: dictionary with weights of layers in potential energy network
        :arg T_kin_layer_weights: dictionary with weights of layers in kinetic energy network
        """
        if V_pot_layer_weights:
            for layer in self.V_pot_layers:
                if layer.name in V_pot_layer_weights.keys():
                    layer.set_weights(V_pot_layer_weights[layer.name])
                else:
                    print(f"WARNING: Weights of V_pot layer '{layer.name}' not set.")
            if "final" in V_pot_layer_weights.keys():
                self.V_pot_final_layer.set_weights(V_pot_layer_weights["final"])
            else:
                print("WARNING: Weights of final V_pot layer not set.")
        if T_kin_layer_weights:
            for layer in self.T_kin_layers:
                if layer.name in T_kin_layer_weights.keys():
                    layer.set_weights(T_kin_layer_weights[layer.name])
                else:
                    print(f"WARNING: Weights of T_kin layer '{layer.name}' not set.")
            if "final" in T_kin_layer_weights.keys():
                self.T_kin_final_layer.set_weights(T_kin_layer_weights["final"])
            else:
                print("WARNING: Weights of final T_kin layer not set.")

    def V_pot(self, q):
        """Evaluate potential energy network V(q)

        :arg q: position q  which to evaluate the potential
        """
        x = q
        for layer in self.V_pot_layers:
            x = layer(x)
        x = self.V_pot_final_layer(x)
        return x

    def T_kin(self, p):
        """Evaluate kinetic energy network T(p)

        :arg p: momentum p at which to evaluate the kinetic energy
        """
        x = p
        for layer in self.T_kin_layers:
            x = layer(x)
        x = self.T_kin_final_layer(x)
        return x

    @tf.function
    def step(self, q_n, p_n):
        """Carry out a single Stoermer-Verlet step

        This function maps (q_n,p_n) to (q_{n+1},p_{n+1}) using a single Stoermer
        Verlet step

        :arg q_n: current position q_n
        :arg p_n: current momentum p_n
        """
        # p_{n+1/2} = p_n - dt/2*dV/dq(q_n)
        dV_dq = tf.gradients(self.V_pot(q_n), q_n)[0]
        p_n = p_n - 0.5 * self.dt * dV_dq

        # q_{n+1} = q_n + dt*dT/dq(p_{n+1/2})
        dT_dp = tf.gradients(self.T_kin(p_n), p_n)[0]
        q_n = q_n + self.dt * dT_dp

        # p_{n+1} = p_{n+1/2} - dt/2*dV/dq(q_{n+1})
        dV_dq = tf.gradients(self.V_pot(q_n), q_n)[0]
        p_n = p_n - 0.5 * self.dt * dV_dq

        return q_n, p_n


class StrangSplittingModel(SymplecticModel):
    """Single step of a Symplectic integrator update for a general system with Hamiltonian H(q,p)

    Implements the symplectic Strang splitting method described in [1]

    [1]: Molei Tao: "Explicit symplectic approximation of nonseparable Hamiltonians:
         algorithm and long time performance" https://arxiv.org/abs/1609.02212

    For this, the Hamiltonian H(x,p) is eqtended to

      Hbar(q,p,x,y) = H_A(q,y) + H_B(z,p) + omega*H_C(q,p,x,y)

    where H_A(q,y) = H(q,y), H_B(x,p) = H(x,p) and
    H_C(q,p,x,y) = 1/2 * ( ||q-x||_2^2 + ||p-y||_2^2 )

    The integrator corresponds to the following symmetric update:

      phi^{dt} = phi_{H_A}^{dt/2} * phi_{H_B}^{dt/2}
               * phi_{omega*H_C}^{dt}
               * phi_{H_B}^{dt/2} * phi_{H_A}^{dt/2}

    where phi_{H_X}^{dt} corresponds to time evolution with the Hamiltonian H_X.

    For H_A, H_B and omega*H_C the action of these operators is written down in Eq. (1)
    of [1].

    The Hamiltonian H(q,p) is represented by a neural network.
    """

    def __init__(self, dim, dt, H_layers, omega=1.0):
        """Construct new instance

        :arg dim: dimension d of the Hamiltonian system
        :arg dt: timestep size
        :arg H_layers: layers encoding the neural network for the Hamiltonian H(q,p)
        :arg omega: scaling parameter of coupling Hamiltonian
        """
        super().__init__(dim, dt)
        self.H_layers = H_layers
        self.H_final_layer = keras.layers.Dense(1, use_bias=False)
        self.omega = omega

    def set_weights(self, H_layer_weights):
        """Set weights of layers in energy networks

        :arg H_layer_weights: dictionary with weights of layers in Hamiltonian network
        """
        if H_layer_weights:
            for layer in self.H_layers:
                if layer.name in H_layer_weights.keys():
                    layer.set_weights(H_layer_weights[layer.name])
                else:
                    print(f"WARNING: Weights of H layer '{layer.name}' not set.")
            if "final" in H_layer_weights.keys():
                self.H_final_layer.set_weights(H_layer_weights["final"])
            else:
                print("WARNING: Weights of final H layer not set.")

    def Hamiltonian(self, q, p):
        """Evaluate Hamiltonian H(q,p)

        :arg q: position at which to evaluate the Hamiltonian
        :arg p: canonical momentum at which to evaluate the Hamiltonian
        """
        x = keras.layers.Concatenate(axis=-1)([q, p])
        for layer in self.H_layers:
            x = layer(x)
        x = self.H_final_layer(x)
        return x

    @tf.function
    def step(self, q_n, p_n):
        """Carry out a single Stoermer-Verlet step

        This function maps (q_n,p_n) to (q_{n+1},p_{n+1}) using a single Stoermer
        Verlet step

        :arg q_n: current position q_n
        :arg p_n: current momentum p_n
        """
        cos_2omega_dt = np.cos(2.0 * self.omega * self.dt)
        sin_2omega_dt = np.sin(2.0 * self.omega * self.dt)
        x_n = q_n
        y_n = p_n
        # **** H_A update ****
        dH_dq, dH_dy = tf.gradients(self.Hamiltonian(q_n, y_n), [q_n, y_n])
        x_n = x_n + 0.5 * self.dt * dH_dy
        p_n = p_n - 0.5 * self.dt * dH_dq
        # **** H_B update ****
        dH_dx, dH_dp = tf.gradients(self.Hamiltonian(x_n, p_n), [x_n, p_n])
        q_n = q_n + 0.5 * self.dt * dH_dp
        y_n = y_n - 0.5 * self.dt * dH_dx
        # **** H_C update ****
        q_n, p_n, x_n, y_n = (
            (
                (1 + cos_2omega_dt) * q_n
                + sin_2omega_dt * p_n
                + (1 - cos_2omega_dt) * x_n
                - sin_2omega_dt * y_n
            )
            / 2,
            (
                -sin_2omega_dt * q_n
                + (1 + cos_2omega_dt) * p_n
                + sin_2omega_dt * x_n
                + (1 - cos_2omega_dt) * y_n
            )
            / 2,
            (
                (1 - cos_2omega_dt) * q_n
                - sin_2omega_dt * p_n
                + (1 + cos_2omega_dt) * x_n
                + sin_2omega_dt * y_n
            )
            / 2,
            (
                sin_2omega_dt * q_n
                + (1 - cos_2omega_dt) * p_n
                - sin_2omega_dt * x_n
                + (1 + cos_2omega_dt) * y_n
            )
            / 2,
        )

        # **** H_B update ****
        dH_dx, dH_dp = tf.gradients(self.Hamiltonian(x_n, p_n), [x_n, p_n])
        q_n = q_n + 0.5 * self.dt * dH_dp
        y_n = y_n - 0.5 * self.dt * dH_dx
        # **** H_A update ****
        dH_dq, dH_dy = tf.gradients(self.Hamiltonian(q_n, y_n), [q_n, y_n])
        x_n = x_n + 0.5 * self.dt * dH_dy
        p_n = p_n - 0.5 * self.dt * dH_dq
        return q_n, p_n
