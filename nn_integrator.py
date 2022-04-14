"""Timesteppers based on neural networks"""

import json
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
import auxilliary
from models import VerletModel


class NNIntegrator(object):
    """Base class for neural network based integrators

    :arg dynamical_system: dynamical system used for integration
    :arg dt: timestep size
    :arg nsteps: number of multisteps
    :arg learning_rate: learning rate
    """

    def __init__(self, dynamical_system, dt, nsteps, learning_rate=1.0e-4):
        self.dynamical_system = dynamical_system
        self.dt = dt
        self.dim = 2 * self.dynamical_system.dim
        self.nsteps = nsteps
        self.xp = np.zeros((1, self.nsteps, self.dim))
        self.learning_rate = learning_rate

    @classmethod
    def from_model(cls, dynamical_system, dt, model, learning_rate=1.0e-4):
        """Construct integrator from model

        :arg dynamical_system: underlying dynamical system
        :arg dt: timestep size
        :arg model: neural network model
        """
        nsteps = model.input_shape[1]
        nn_integrator = cls(dynamical_system, dt, nsteps, learning_rate)
        nn_integrator.model = model
        return nn_integrator

    def set_state(self, x, p):
        """Set the current state of the integrator

        :arg x: Array of size nsteps x dim with initial positions
        :arg v: Array of size nsteps x dim with initial velocities
        """
        self.xp[0, :, : self.dim // 2] = x[:, :]
        self.xp[0, :, self.dim // 2 :] = p[:, :]

    @property
    def x(self):
        """Return the current position vector (as a d-dimensional array)"""
        return self.xp[0, -1, : self.dim // 2]

    @property
    def p(self):
        """Return the current velocity vector (as a d-dimensional array)"""
        return self.xp[0, -1, self.dim // 2 :]

    def integrate(self, n_steps):
        """Carry out a given number of integration steps

        :arg n_steps: number of integration steps
        """
        for _ in range(n_steps):
            x_pred = np.asarray(self.model.predict(self.xp)).flatten()
            self.xp = np.roll(self.xp, -1, axis=1)
            self.xp[0, -1, :] = x_pred[:]

    def energy(self):
        """Compute energy of dynamical system at current state"""
        return self.dynamical_system.energy(self.x, self.p)


class MultistepNNIntegrator(NNIntegrator):
    """Multistep integrator. Use a neural network to predict the next state, given
    a number of previous states

    :arg dynamical_system: dynamical system used for integration
    :arg dt: timestep size
    :arg nsteps: Number of steps of the timestepping method
    :arg dense_layers: neural network layers used to predict the next state
    :arg learning_rate: learning rate
    """

    def __init__(
        self, dynamical_system, dt, nsteps, dense_layers, learning_rate=1.0e-4
    ):
        super().__init__(dynamical_system, dt, nsteps, learning_rate)
        self.dim = 2 * self.dynamical_system.dim
        self.dense_layers = dense_layers
        # Build model
        inputs = keras.Input(shape=(self.nsteps, self.dim))
        q_n = tf.unstack(inputs, axis=1)[-1]
        output_layer = keras.layers.Dense(self.dim)
        x = inputs
        for layer in dense_layers:
            x = layer(x)
        x = output_layer(x)
        x = keras.layers.Rescaling(self.dt)(x)
        outputs = keras.layers.Add()([q_n, x])
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            loss="mse",
            metrics=[],
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
        )
        self.xp = np.zeros((1, self.nsteps, self.dim))


class HamiltonianVerletNNIntegrator(NNIntegrator):
    """Neural network integrator based on the Hamiltonian Stoermer-Verlet update"""

    def __init__(
        self,
        dynamical_system,
        dt,
        V_pot_layers,
        T_kin_layers,
        V_pot_layer_weights=None,
        T_kin_layer_weights=None,
        learning_rate=1.0e-4,
    ):
        super().__init__(dynamical_system, dt, 1, learning_rate)
        self.V_pot_layers = V_pot_layers
        self.T_kin_layers = T_kin_layers
        self.model = self.build_model(
            self.dim,
            self.dt,
            self.V_pot_layers,
            self.T_kin_layers,
            V_pot_layer_weights,
            T_kin_layer_weights,
            self.learning_rate,
        )

    @staticmethod
    def build_model(
        dim,
        dt,
        V_pot_layers,
        T_kin_layers,
        V_pot_layer_weights=None,
        T_kin_layer_weights=None,
        learning_rate=1.0e-4,
    ):
        """Build underlying Verlet model

        :arg dim: dimension of dynamical system
        :arg dt: timestep size
        :arg V_pot_layers: Layers used for potential energy network
        :arg T_kin_layers: Layers used for kinetic energy network
        :arg V_pot_layer_weights: list of layer weights for potential energy network
        :arg T_kin_layer_weights: list of layer weights for kinetic energy network
        :arg learning_rate: learning rate
        """
        inputs = keras.Input(shape=(1, dim))
        verlet_model = VerletModel(dim, dt, V_pot_layers, T_kin_layers)
        outputs = verlet_model(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.build(input_shape=(None, 1, dim))
        verlet_model.set_weights(V_pot_layer_weights, T_kin_layer_weights)
        model.compile(
            loss="mse",
            metrics=[],
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        )
        return model

    def save_layers(self, layers, final_layer, filename):
        """Save list of layers in a given file

        :arg layers: list of (intermediate) layers to save
        :arg final_layer: final layer to save (only weights)
        :arg filename: name of file to save to
        """
        layer_list = []
        for layer in layers:
            layer_list.append(
                {
                    "class": type(layer).__module__ + "." + type(layer).__name__,
                    "config": layer.get_config(),
                    "weights": layer.get_weights(),
                }
            )
        layer_list.append(final_layer.get_weights())
        with open(filename, "w", encoding="utf8") as f:
            json.dump(layer_list, f, cls=auxilliary.ndarrayEncoder, indent=4)

    @classmethod
    def load_layers(cls, filename):
        """Load a list of layers from a given file.
        Returns a list of layers and the associated weights.

        :arg filename: name of file to read
        """
        # >>> import keras <<<

        layers = []
        layer_weights = {}
        with open(filename, "r", encoding="utf8") as f:
            layer_list = json.load(f, cls=auxilliary.ndarrayDecoder)
        for layer_dict in layer_list[:-1]:
            layer_cls = eval(layer_dict["class"])
            config = layer_dict["config"]
            weights = layer_dict["weights"]
            layer = layer_cls.from_config(config)
            layer_weights[layer.name] = weights
            layers.append(layer)
        layer_weights["final"] = layer_list[-1]
        return layers, layer_weights

    def save_model(self, dirname):
        """Save Hamiltonian model to disk

        This saves the two sequential models for the potential- and kinetic energy as well
        as the two model parameters (dimension dim and timestep size dt)

        :arg dirname: Name of directory to save model to
        """
        specs = {"dt": self.dt, "dim": self.dim}
        shutil.rmtree(dirname, ignore_errors=True)
        os.mkdir(dirname)
        with open(dirname + "/specifications.json", "w", encoding="utf8") as f:
            json.dump(specs, f, ensure_ascii=True)
        verlet_model = self.model.layers[-1]
        self.save_layers(
            self.V_pot_layers,
            verlet_model.V_pot_final_layer,
            dirname + "/V_pot_layers.json",
        )
        self.save_layers(
            self.T_kin_layers,
            verlet_model.T_kin_final_layer,
            dirname + "/T_kin_layers.json",
        )

    @classmethod
    def load_model(cls, dirname, new_dt=None, learning_rate=1.0e-4):
        """Load Hamiltonian model from disk

        :args dirname: directory containing the model specifications
        :arg new_dt: new value for the timestep size
        :arg learning_rate: learning rate
        """
        with open(dirname + "/specifications.json", encoding="utf8") as f:
            specs = json.load(f)
        V_pot_layers, V_pot_layer_weights = cls.load_layers(
            dirname + "/V_pot_layers.json"
        )
        T_kin_layers, T_kin_layer_weights = cls.load_layers(
            dirname + "/T_kin_layers.json"
        )
        dim, dt = specs["dim"], specs["dt"]
        if new_dt:
            dt = new_dt
        model = cls.build_model(
            dim,
            dt,
            V_pot_layers,
            T_kin_layers,
            V_pot_layer_weights,
            T_kin_layer_weights,
            learning_rate,
        )
        return model
