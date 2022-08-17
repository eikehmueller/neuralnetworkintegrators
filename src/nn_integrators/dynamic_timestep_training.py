"""Support for training with timestep size that changes

Provides a callback that adjusts the timestep size of the neural network
integrator during training.
"""

import tensorflow as tf
from tensorflow import keras


class DynamicTimestepCallback(keras.callbacks.Callback):
    """Callback for dynamically adjusting the timestep size during training

    Initialised with a dictionary with key-value pairs of the form {minimal epoch -> timestep size}.
    At the beginning of each epoch, the callback goes through all entries of this dictionary and if
    the current epoch is at least a given minimal epoch, the size of the timestep for the neural
    network integrator and the data generator will be set to a specified value.
    """

    def __init__(self, data_generator, timestep_schedule, log_dir):
        """Construct new instance

        :arg data_generator: Generator class for data
        :arg timestep-schedule: A dictionary with key-value pair of the form
                                {minimal epoch -> timestep size}
        :arg_dir: Directory for writing out the the current timestep
        """
        self.data_generator = data_generator
        self.timestep_schedule = timestep_schedule
        # Check that all timesteps are multiples of the training timestep
        dt_train = self.data_generator.train_integrator.dt
        for dt in self.timestep_schedule.values():
            assert abs(round(dt / dt_train) * dt_train / dt - 1.0) < 1e-12
        self.timestep_writer = tf.summary.create_file_writer(log_dir)

    def on_train_begin(self, logs=None):  # pylint: disable=unused-argument
        """Check that the model in the data generator is used for training"""
        assert self.data_generator.nn_integrator.model is self.model

    def on_epoch_begin(self, epoch, logs=None):  # pylint: disable=unused-argument
        """Check which timestep size should be used"""
        for min_epoch, dt in self.timestep_schedule.items():
            if epoch >= min_epoch:
                self.data_generator.nn_integrator.dt = dt

    def on_epoch_end(self, epoch, logs=None):  # pylint: disable=unused-argument
        """Check which timestep size should be used"""
        with self.timestep_writer.as_default():
            tf.summary.scalar(
                "timestep size", self.data_generator.nn_integrator.dt, step=epoch
            )


def create_linear_timestep_schedule(dt_train, dt_min, dt_max, nepoch, ninc):
    """Create a dictionary that can be used by DynamicTimestepCallback.
    The size of the minimal timestep increases linearly with the epoch, varying
    between dt_min and dt_max, such that it is always a multiple of dt_train.
    If will only change every ninc epochs, however.

    :arg dt_train: timestep size of training integrator
    :arg dt_min: minimal timestep size
    :arg dt_max: maximal timestep size
    :arg nepoch: largest epoch to generator timestep for
    :arg ninc: increment for created epochs.
    """
    schedule = {}
    for epoch in range(0, nepoch + 1, ninc):
        dt = dt_min + (dt_max - dt_min) * epoch / nepoch
        n = int(dt / dt_train)
        schedule[epoch] = n * dt_train
    return schedule
