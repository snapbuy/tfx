# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Supplement for penguin_utils_base.py with specifics for Flax models.

Using TFX with Flax instead of Keras requires only a few changes for the
Trainer component. However, Flax is simpler than Keras and does not come
pre-packaged with a training loop, model saving, and other bells and whistles,
hence the training code with Flax is more verbose. This file contains the
definition of the Flax model for Penguin dataset, the training loop, and the
model creation and saving.

The code in this file is structures in three parts: Part 1 contains
standard Flax code to define and train the model, independent of any TFX
specifics; Part 2 contains the conversion of the Flax model to a tf.saved_model,
using jax2tf, but also independent of TFX; Part 3 contains the customization of
TFX components.
"""
import functools
from typing import Dict, Iterator, List, Tuple

import absl

import flax
from flax import linen as nn
from flax.metrics import tensorboard

import jax
from jax import numpy as jnp
from jax.experimental import jax2tf
import numpy as np

import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.examples.penguin import penguin_utils_base as base

# The transformed feature names
_FEATURE_KEYS_XF = list(map(base.transformed_name, base.FEATURE_KEYS))

# Type abbreviations: (B is the batch size)
Array = np.ndarray
InputBatch = Dict[str, Array]  # keys are _FEATURE_KEYS_XF and values f32[B, 1]
LogitBatch = Array  # of shape f32[B, 3]
LabelBatch = Array  # of shape int64[B, 1]


### Part 1: Definition of the Flax model and its training loop.
#
# This part is standard Flax code, independent of any TFX specifics.
#
def _get_trained_model(train_data: Iterator[Tuple[InputBatch, LabelBatch]],
                       eval_data: Iterator[Tuple[InputBatch, LabelBatch]],
                       num_epochs: int, steps_per_epoch: int,
                       eval_steps_per_epoch: int, tensorboard_log_dir: str):
  """Execute model training and evaluation loop.

  Args:
    train_data: an iterator of pairs with training data.
    eval_data: an iterator of pairs with evaluation data.
    num_epochs: number of training epochs. Should cover all `train_data`.
    steps_per_epoch: number of steps for a training epoch.
    eval_steps_per_epoch: number of steps for evaluation. Should cover all
      `eval_data`.
    tensorboard_log_dir: Directory where the tensorboard summaries are written.

  Returns:
    An instance of tf.Model.
  """
  learning_rate = 1e-2

  rng = jax.random.PRNGKey(0)

  summary_writer = tensorboard.SummaryWriter(tensorboard_log_dir)
  summary_writer.hparams(
      dict(
          learning_rate=learning_rate,
          num_epochs=num_epochs,
          steps_per_epoch=steps_per_epoch,
          eval_steps_per_epoch=eval_steps_per_epoch))

  rng, init_rng = jax.random.split(rng)
  # Initialize with some fake data of the proper shape.
  init_val = dict((feature, jnp.array([[1.]], dtype=jnp.float32))
                  for feature in _FEATURE_KEYS_XF)
  model = _FlaxPenguinModel()
  params = model.init(init_rng, init_val)['params']

  optimizer_def = flax.optim.Adam(learning_rate=learning_rate)
  optimizer = optimizer_def.create(params)

  for epoch in range(1, num_epochs + 1):
    optimizer, train_metrics = _train_epoch(model, optimizer, train_data,
                                            steps_per_epoch)
    absl.logging.info('Flax train epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
                      train_metrics['loss'], train_metrics['accuracy'] * 100)

    eval_metrics = _eval_epoch(model, optimizer.target, eval_data,
                               eval_steps_per_epoch)
    absl.logging.info('Flax eval epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
                      eval_metrics['loss'], eval_metrics['accuracy'] * 100)
    summary_writer.scalar('train_loss', train_metrics['loss'], epoch)
    summary_writer.scalar('train_accuracy', train_metrics['accuracy'], epoch)
    summary_writer.scalar('eval_loss', eval_metrics['loss'], epoch)
    summary_writer.scalar('eval_accuracy', eval_metrics['accuracy'], epoch)

  summary_writer.flush()

  # The prediction function for the trained model
  def predict(params, inputs):
    return model.apply({'params': params}, inputs)

  trained_params = optimizer.target
  return _SavedModelWrapper.from_flax_prediction(predict, trained_params)


class _FlaxPenguinModel(nn.Module):
  """The model definition."""

  @nn.compact
  def __call__(self, x_dict: InputBatch) -> LogitBatch:
    # Each feature is of shape f32[B, 1]
    x_tuple = tuple(x_dict[feature] for feature in _FEATURE_KEYS_XF)
    x_array = jnp.concatenate(x_tuple, axis=-1)  # shape: [B, 4]
    assert x_array.ndim == 2
    assert x_array.shape[1] == 4
    x = x_array

    x = nn.Dense(features=8)(x)
    x = nn.relu(x)
    x = nn.Dense(features=8)(x)
    x = nn.relu(x)
    x = nn.Dense(features=3)(x)
    x = nn.log_softmax(x, axis=-1)
    return x


def _train_epoch(model, optimizer, train_data, steps_per_epoch):
  """Train for a single epoch."""
  batch_metrics = []
  for _ in range(steps_per_epoch):
    inputs, labels = next(train_data)
    optimizer, metrics = _train_step(model, optimizer, inputs, labels)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  epoch_metrics_np = _mean_epoch_metrics(jax.device_get(batch_metrics))
  return optimizer, epoch_metrics_np


def _eval_epoch(model, params, eval_data, steps_per_epoch):
  """Validate for a single epoch."""
  batch_metrics = []
  for _ in range(steps_per_epoch):
    inputs, labels = next(eval_data)
    metrics = _eval_step(model, params, inputs, labels)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  epoch_metrics_np = _mean_epoch_metrics(jax.device_get(batch_metrics))
  return epoch_metrics_np


@functools.partial(jax.jit, static_argnums=0)
def _train_step(model, optimizer, inputs: InputBatch, labels: LabelBatch):
  """Train for a single step, given a batch of inputs and labels."""

  def loss_fn(params):
    logits = model.apply({'params': params}, inputs)
    loss = _categorical_cross_entropy_loss(logits, labels)
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  metrics = _compute_metrics(logits, labels)
  return optimizer, metrics


@functools.partial(jax.jit, static_argnums=0)
def _eval_step(model, params, inputs: InputBatch, labels: LabelBatch):
  logits = model.apply({'params': params}, inputs)
  return _compute_metrics(logits, labels)


def _categorical_cross_entropy_loss(logits: LogitBatch, labels: LabelBatch):
  # assumes that the logits use log_softmax activations.
  onehot_labels = (labels == jnp.arange(3)[None]).astype(jnp.float32)
  # onehot_labels: f32[B, 3]
  z = -jnp.sum(onehot_labels * logits, axis=-1)  # f32[B]
  return jnp.mean(z)  # f32


def _compute_metrics(logits: LogitBatch, labels: LabelBatch):
  # assumes that the logits use log_softmax activations.
  loss = _categorical_cross_entropy_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels[..., 0])
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


def _mean_epoch_metrics(
    batch_metrics: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      metric_name:
      np.mean([metrics[metric_name] for metrics in batch_metrics_np])
      for metric_name in batch_metrics_np[0]
  }
  return epoch_metrics_np


### Part 2: Wrap the Flax model as a tf.saved_model.
#
# This part is independent of any TFX specifics.
#
class _SavedModelWrapper(tf.train.Checkpoint):
  """Wraps a function and its parameters for saving to a SavedModel.

  Implements the interface described at
  https://www.tensorflow.org/hub/reusable_saved_models.

  This class contains all the code needed to convert a Flax model to a
  TensorFlow saved model.
  """

  @classmethod
  def from_flax_prediction(cls, predict_fn, trained_params):

    tf_fn = jax2tf.convert(predict_fn, with_gradient=False, enable_xla=True)

    # Create tf.Variables for the parameters. If you want more useful variable
    # names, you can use `tree.map_structure_with_path` from the `dm-tree`
    # package.
    param_vars = tf.nest.map_structure(
        # Due to a bug in SavedModel it is not possible to use tf.GradientTape
        # on a function converted with jax2tf and loaded from SavedModel.
        # Thus, we mark the variables as non-trainable to ensure that users of
        # the SavedModel will not try to fine tune them.
        lambda param: tf.Variable(param, trainable=False),
        trained_params)
    tf_graph = tf.function(
        lambda inputs: tf_fn(param_vars, inputs),
        autograph=False,
        experimental_compile=True)
    model = _SavedModelWrapper(tf_graph, param_vars)
    return model

  def __init__(self, tf_graph, param_vars):
    """Builds the tf.Module.

    Args:
      tf_graph: a tf.function taking one argument (the inputs), which can be be
        tuples/lists/dictionaries of np.ndarray or tensors. The function may
        have references to the tf.Variables in `param_vars`.
      param_vars: the parameters, as tuples/lists/dictionaries of tf.Variable,
        to be saved as the variables of the SavedModel.
    """
    super().__init__()
    # Implement the interface from
    # https://www.tensorflow.org/hub/reusable_saved_models
    self.variables = tf.nest.flatten(param_vars)
    self.trainable_variables = [v for v in self.variables if v.trainable]
    self._tf_graph = tf_graph

  @tf.function
  def __call__(self, inputs):
    return self._tf_graph(inputs)

  def save(self, model_dir, *, signatures):
    """Saves the model.

    The only reason we have this methos such that an instance of
    _SavedModelWrapper can be saved using an invocation similar to how a
    Keras model is saved in `penguin_utils.py`.
    Args:
      model_dir: the directory where to save the model.
      signatures: Signatures to save with the SavedModel. Please see the
        `signatures` argument in `tf.saved_model.save` for details.
    """
    tf.saved_model.save(self, model_dir, signatures=signatures)

### Part 2: Customization of TFX components

# TFX Transform will call this function.
preprocessing_fn = base.preprocessing_fn


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = base.input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      batch_size=base.TRAIN_BATCH_SIZE)

  eval_dataset = base.input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      tf_transform_output,
      batch_size=base.EVAL_BATCH_SIZE)

  # TODO(necula): Use the hyperparameters to construct the model
  model = _get_trained_model(
      train_dataset.as_numpy_iterator(),
      eval_dataset.as_numpy_iterator(),
      num_epochs=1,
      steps_per_epoch=fn_args.train_steps,
      eval_steps_per_epoch=fn_args.eval_steps,
      tensorboard_log_dir=fn_args.model_run_dir)
  # TODO(necula): batch polymorphic model not yet supported.
  serving_batch_size = 1

  signatures = {
      'serving_default':
          base.get_serve_tf_examples_fn(
              model, tf_transform_output).get_concrete_function(
                  tf.TensorSpec(
                      shape=[serving_batch_size],
                      dtype=tf.string,
                      name='examples')),
  }
  model.save(fn_args.serving_model_dir, signatures=signatures)
