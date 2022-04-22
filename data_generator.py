import numpy as np
import tensorflow as tf


class DataGenerator(object):
    """Class for generating data for training Neural Network integrators

    Generates trajectories that can be used for training,

    x_0,x_1,...,x_{s-1},x_s

    for random initial conditions x_0, where x_j approximates the solution at time j*dt and
    dt is the (large) timestep size of the neural network integrator that we want to train.
    The trajectories are generated with a training integrator that uses a smaller timestep
    size dt_{train} << dt. Returns tensors (X,y) where X is of size s x d and y is of size d,
    with d being the phase-space dimension of the dynamical system.
    """

    def __init__(self, nn_integrator, train_integrator):
        """Construct a new instance

        :arg nn_integrator: neural network time integrator
        :arg train_integrator: training integrator
        """
        self.nn_integrator = nn_integrator
        self.train_integrator = train_integrator
        self.dynamical_system = self.train_integrator.dynamical_system
        self.phasespace_dim = self.nn_integrator.model.input_spec[0].shape[-1]
        self.dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(self.nn_integrator.nsteps, self.phasespace_dim),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(shape=(self.phasespace_dim), dtype=tf.float32),
            ),
        )

    def _generator(self):
        """Generate a new data sample (X,y)"""
        dim = self.dynamical_system.dim
        state = np.zeros((self.nn_integrator.nsteps + 1, self.phasespace_dim))
        extended_phasespace = self.phasespace_dim == 4 * dim
        while True:
            self.dynamical_system.set_random_state(
                state[0, 0:dim], state[0, dim : 2 * dim]
            )
            self.train_integrator.set_state(state[0, 0:dim], state[0, dim : 2 * dim])
            if extended_phasespace:
                self.dynamical_system.set_random_state(
                    state[0, 2 * dim : 3 * dim], state[0, 3 * dim : 4 * dim]
                )
                self.train_integrator.set_extended_state(
                    state[0, 2 * dim : 3 * dim], state[0, 3 * dim : 4 * dim]
                )

            for k in range(self.nn_integrator.nsteps):
                self.train_integrator.integrate(
                    int(self.nn_integrator.dt / self.train_integrator.dt)
                )
                state[k + 1, 0:dim] = self.train_integrator.q[:]
                state[k + 1, dim : 2 * dim] = self.train_integrator.p[:]
                if extended_phasespace:
                    state[k + 1, 2 * dim : 3 * dim] = self.train_integrator.x[:]
                    state[k + 1, 3 * dim : 4 * dim] = self.train_integrator.y[:]
            X = state[:-1, :]
            y = state[-1, :]
            yield (X, y)
