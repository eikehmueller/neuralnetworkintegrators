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

    The initial state that is used to start the trajectories is drawn at random after
    random_reset_interval steps. If this parameter is set to 1, then each trajectory
    starts at a randomly chosen point in phase space.
    """

    def __init__(self, nn_integrator, train_integrator, random_reset_interval=1):
        """Construct a new instance

        :arg nn_integrator: neural network time integrator
        :arg train_integrator: training integrator
        :arg random_reset_interval: number of integrator steps after which
            the initial state is reset to a random value
        """
        self.nn_integrator = nn_integrator
        self.train_integrator = train_integrator
        self.random_reset_interval = random_reset_interval
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

            # Draw new random initial start
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
            # Now loop over timesteps
            for k in range(1, self.nn_integrator.nsteps + self.random_reset_interval):
                self.train_integrator.integrate(
                    int(self.nn_integrator.dt / self.train_integrator.dt)
                )
                j = min(self.nn_integrator.nsteps, k)
                state[j, 0:dim] = self.train_integrator.q[:]
                state[j, dim : 2 * dim] = self.train_integrator.p[:]
                if extended_phasespace:
                    state[j, 2 * dim : 3 * dim] = self.train_integrator.x[:]
                    state[j, 3 * dim : 4 * dim] = self.train_integrator.y[:]
                if k >= self.nn_integrator.nsteps:
                    X = state[:-1, :]
                    y = state[-1, :]
                    yield (X, y)
                    state = np.roll(state, -1, axis=0)
