"""Timesteppers based on neural networks"""

import json
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
import auxilliary
from models import StrangSplittingModel, VerletModel


class NNIntegrator(object):
    """Base class for neural network based integrators

    :arg dynamical_system: dynamical system used for integration
    :arg dt: timestep size
    :arg nsteps: number of multisteps
    :arg learning_rate: learning rate
    """

    def __init__(self, dynamical_system, dt, nsteps, learning_rate=1.0e-4):
        self.dynamical_system = dynamical_system
        self._dt = dt
        self.dim = 2 * self.dynamical_system.dim
        self.nsteps = nsteps
        self.qp = np.zeros((1, self.nsteps, self.dim))
        self.learning_rate = learning_rate
        self.model = None

    @property
    def dt(self):
        """Timestep size"""
        return self._dt

    @dt.setter
    def dt(self, new_dt):
        """Set timestep size

        :arg new_dt: new value of timestep size"""
        self._dt = new_dt

    def set_state(self, q, p):
        """Set the current state of the integrator

        :arg q: Array of size nsteps x dim with initial positions
        :arg p: Array of size nsteps x dim with initial momenta
        """
        self.qp[0, :, : self.dim // 2] = q[:, :]
        self.qp[0, :, self.dim // 2 :] = p[:, :]

    @property
    def q(self):
        """Return the current position vector (as a d-dimensional array)"""
        return self.qp[0, -1, : self.dim // 2]

    @property
    def p(self):
        """Return the current velocity vector (as a d-dimensional array)"""
        return self.qp[0, -1, self.dim // 2 :]

    def integrate(self, n_steps):
        """Carry out a given number of integration steps

        :arg n_steps: number of integration steps
        """
        for _ in range(n_steps):
            q_pred = np.asarray(self.model.predict(self.qp)).flatten()
            self.qp = np.roll(self.qp, -1, axis=1)
            self.qp[0, -1, :] = q_pred[:]

    def energy(self):
        """Compute energy of dynamical system at current state"""
        return self.dynamical_system.energy(self.q, self.p)


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
        self, dynamical_system, dt, nsteps, dense_layers=None, learning_rate=1.0e-4
    ):
        super().__init__(dynamical_system, dt, nsteps, learning_rate)
        self.__dt_tf = dt
        self.dim = 2 * self.dynamical_system.dim
        if dense_layers:
            self.dense_layers = dense_layers
        else:
            self.dense_layers = []
        # Build model
        inputs = keras.Input(shape=(self.nsteps, self.dim))
        q_n = tf.unstack(inputs, axis=1)[-1]
        output_layer = keras.layers.Dense(self.dim)
        x = inputs
        for layer in dense_layers:
            x = layer(x)
        x = output_layer(x)
        x = keras.layers.Rescaling(self.__dt_tf)(x)
        outputs = keras.layers.Add()([q_n, x])
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            loss="mse",
            metrics=[],
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
        )
        self.qp = np.zeros((1, self.nsteps, self.dim))

    @property
    def dt(self):
        """Timesteps size"""
        return self._dt

    @dt.setter
    def dt(self, new_dt):
        """Set timestep size

        This also sets the tensorflow timestep of the underlying multistep model

        :arg new_dt: new value of timestep size"""
        self.dt = new_dt
        self.__dt_tf.assign(new_dt)

    @classmethod
    def from_model(cls, dynamical_system, dt, model):
        """Construct integrator from model

        :arg dynamical_system: underlying dynamical system
        :arg dt: timestep size
        :arg model: neural network model
        """
        nsteps = model.input_shape[1]
        nn_integrator = cls(dynamical_system, dt, nsteps)
        nn_integrator.model = model  # pylint: disable=attribute-defined-outside-init
        return nn_integrator


class HamiltonianNNIntegrator(NNIntegrator):
    """Neural network integrator based on a Hamiltonian update"""

    def __init__(
        self,
        dynamical_system,
        dt,
        learning_rate=1.0e-4,
    ):
        """Construct new instance"""
        super().__init__(dynamical_system, dt, 1, learning_rate)

    @property
    def dt(self):
        """Timesteps size"""
        return self._dt

    @dt.setter
    def dt(self, new_dt):
        """Set timestep size.

        This also sets the tensorflow timestep of the underlying model

        :arg new_dt: new value of timestep size"""
        self._dt = new_dt
        # Set the (tensorflow) timestep of the underlying model
        self.model.layers[-1].dt = new_dt

    @classmethod
    def from_model(cls, dynamical_system, dt, model):
        """Construct integrator from model

        :arg dynamical_system: underlying dynamical system
        :arg dt: timestep size
        :arg model: neural network model
        """
        nn_integrator = cls(dynamical_system, dt)
        nn_integrator.model = model  # pylint: disable=attribute-defined-outside-init
        return nn_integrator

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

    def save_specification(self, dirname):
        """Save model specifications to file

        Write the file specifications.json in a specified directory

        :arg dirname: model directory
        """
        specs = {"dt": self.dt, "dim": self.dim}
        shutil.rmtree(dirname, ignore_errors=True)
        os.mkdir(dirname)
        with open(dirname + "/specifications.json", "w", encoding="utf8") as f:
            json.dump(specs, f, ensure_ascii=True)

    @classmethod
    def load_layers(cls, filename):
        """Load a list of layers from a given file.
        Returns a list of layers and the associated weights.

        :arg filename: name of file to read
        """
        import keras  # pylint: disable=reimported,redefined-outer-name,unused-import,import-outside-toplevel

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

        This saves the model to disk

        :arg dirname: Name of directory to save model to
        """

    @classmethod
    def load_model(cls, dirname, learning_rate=1.0e-4):
        """Load Hamiltonian model from disk

        :args dirname: directory containing the model specifications
        :arg learning_rate: learning rate
        """


class HamiltonianVerletNNIntegrator(HamiltonianNNIntegrator):
    """Neural network integrator based on the Hamiltonian Stoermer-Verlet update"""

    def __init__(
        self,
        dynamical_system,
        dt,
        V_pot_layers=None,
        T_kin_layers=None,
        V_pot_layer_weights=None,
        T_kin_layer_weights=None,
        learning_rate=1.0e-4,
    ):
        super().__init__(dynamical_system, dt, learning_rate)
        if V_pot_layers:
            self.V_pot_layers = V_pot_layers
        else:
            self.V_pot_layers = []
        if T_kin_layers:
            self.T_kin_layers = T_kin_layers
        else:
            self.T_kin_layers = []
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

    def save_model(self, dirname):
        """Save Hamiltonian model to disk

        This saves the two sequential models for the potential- and kinetic energy as well
        as the two model parameters (dimension dim and timestep size dt)

        :arg dirname: Name of directory to save model to
        """
        self.save_specification(dirname)
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
    def load_model(cls, dirname, learning_rate=1.0e-4):
        """Load Hamiltonian model from disk

        :args dirname: directory containing the model specifications
        :arg learning_rate: learning rate
        """
        with open(dirname + "/specifications.json", encoding="utf8") as f:
            specs = json.load(f)
        dim, dt = specs["dim"], specs["dt"]
        V_pot_layers, V_pot_layer_weights = cls.load_layers(
            dirname + "/V_pot_layers.json"
        )
        T_kin_layers, T_kin_layer_weights = cls.load_layers(
            dirname + "/T_kin_layers.json"
        )
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


class HamiltonianStrangSplittingNNIntegrator(HamiltonianNNIntegrator):
    """Neural network integrator based on the Hamiltonian Strang splitting update"""

    def __init__(
        self,
        dynamical_system,
        dt,
        H_layers=None,
        H_layer_weights=None,
        learning_rate=1.0e-4,
    ):
        """Construct new instance

        :arg dynamical_system: Underlying dynamical system
        :arg dt: timestep size
        :arg H_layers: intermediate layers of Hamiltonian neural network
        :arg learning_rate: Learning rate used during training
        """
        super().__init__(dynamical_system, dt, learning_rate)
        if H_layers:
            self.H_layers = H_layers
        else:
            self.H_layers = []
        self.H_layer_weights = H_layer_weights
        self.qpxy = np.zeros((1, 1, 2 * self.dim))  # extended state
        self.model = self.build_model(
            self.dim,
            self.dt,
            self.H_layers,
            self.H_layer_weights,
            self.learning_rate,
        )

    @staticmethod
    def build_model(
        dim,
        dt,
        H_layers,
        H_layer_weights,
        learning_rate=1.0e-4,
    ):
        """Build underlying Strang splitting model

        :arg dim: dimension of dynamical system
        :arg dt: timestep size
        :arg H_layers: Layers used for Hamiltonian network
        :arg learning_rate: learning rate
        """
        inputs = keras.Input(shape=(1, 2 * dim))
        strang_splitting_model = StrangSplittingModel(2 * dim, dt, H_layers)
        outputs = strang_splitting_model(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.build(input_shape=(None, 1, 2 * dim))
        strang_splitting_model.set_weights(H_layer_weights)
        model.compile(
            loss="mse",
            metrics=[],
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        )
        return model

    def set_state(self, q, p):
        """Set the current state of the integrator

        :arg q: Array of size 1 x dim with initial positions
        :arg p: Array of size 1 x dim with initial momenta
        """
        self.qpxy[0, :, : self.dim // 2] = q[:, :]
        self.qpxy[0, :, self.dim // 2 : self.dim] = p[:, :]
        self.qpxy[0, :, self.dim : 3 * self.dim // 2] = q[:, :]
        self.qpxy[0, :, 3 * self.dim // 2 :] = p[:, :]

    def set_extended_state(self, x, y):
        """Set the current extended state of the integrator

        :arg x: Array of size 1 x dim with initial positions in extended phase space
        :arg y: Array of size 1 x dim with initial momenta
        """
        self.qpxy[0, :, self.dim : 3 * self.dim // 2] = x[:, :]
        self.qpxy[0, :, 3 * self.dim // 2 :] = y[:, :]

    @property
    def q(self):
        """Return the current extended position vector (as a d-dimensional array)"""
        return self.qpxy[0, -1, : self.dim // 2]

    @property
    def p(self):
        """Return the current extended velocity vector (as a d-dimensional array)"""
        return self.qpxy[0, -1, self.dim // 2 : self.dim]

    @property
    def x(self):
        """Return the current extended position vector (as a d-dimensional array)"""
        return self.qpxy[0, -1, self.dim : 3 * self.dim // 2]

    @property
    def y(self):
        """Return the current extended velocity vector (as a d-dimensional array)"""
        return self.qpxy[0, -1, 3 * self.dim // 2 :]

    def integrate(self, n_steps):
        """Carry out a given number of integration steps

        :arg n_steps: number of integration steps
        """
        for _ in range(n_steps):
            qpxy_pred = np.asarray(self.model.predict(self.qpxy)).flatten()
            self.qpxy = np.roll(self.qpxy, -1, axis=1)
            self.qpxy[0, -1, :] = qpxy_pred[:]

    def save_model(self, dirname):
        """Save Hamiltonian model to disk

        This saves the sequential models for the Hamiltonian as well
        as the two model parameters (dimension dim and timestep size dt)

        :arg dirname: Name of directory to save model to
        """
        self.save_specification(dirname)
        strang_splitting_model = self.model.layers[-1]
        self.save_layers(
            self.H_layers,
            strang_splitting_model.H_final_layer,
            dirname + "/H_layers.json",
        )

    @classmethod
    def load_model(cls, dirname, learning_rate=1.0e-4):
        """Load Hamiltonian model from disk

        :args dirname: directory containing the model specifications
        :arg learning_rate: learning rate
        """
        with open(dirname + "/specifications.json", encoding="utf8") as f:
            specs = json.load(f)
        dim, dt = specs["dim"], specs["dt"]
        H_layers, H_layer_weights = cls.load_layers(dirname + "/H_layers.json")
        model = cls.build_model(
            dim,
            dt,
            H_layers,
            H_layer_weights,
            learning_rate,
        )
        return model
