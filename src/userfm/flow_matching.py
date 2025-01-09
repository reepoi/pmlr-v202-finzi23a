# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core flow matching model implementation, noise schedule, type, and training."""

import functools
import time
import logging
from typing import Any, Callable, Sequence, Union

from flax.core.frozen_dict import FrozenDict
from flax import linen as nn
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import optax

from userfm import cs, optimal_transport

Scorefn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
PRNGKey = jnp.ndarray
TimeType = Union[float, jnp.ndarray]
ArrayShape = Sequence[int]
ParamType = Any


log = logging.getLogger(__file__)


def nonefn(x):  # pylint: disable=unused-argument
    return None


class ModelDebug(nn.Module):
    @nn.compact
    def __call__(self, x, t, train=True, cond=None):
        x = x.squeeze(2)
        t = t[:, None]
        a = jnp.concat((x, t), axis=1)
        hidden_dim = 64
        a = nn.Dense(features=hidden_dim, name='dense0')(a)
        a = nn.elu(a)
        a = nn.Dense(features=hidden_dim, name='dense1')(a)
        a = nn.elu(a)
        a = nn.Dense(features=hidden_dim, name='dense2')(a)
        a = nn.elu(a)
        a = nn.Dense(features=hidden_dim, name='dense3')(a)
        a = nn.elu(a)
        a = nn.Dense(features=hidden_dim, name='dense4')(a)
        a = nn.elu(a)
        a = nn.Dense(features=1, name='dense5')(a)
        return a[..., None]


def train_flow_matching(
    cfg,
    model,
    dataloader,
    data_std,
    cond_fn=lambda _: None,  # function: array -> array or None
    num_ema_foldings=5,  # Taos: todo: add this to config
    writer=None,
    report=None,
    ckpt=None,
    key=None,
    rng_seed=None,  # to avoid initing jax
):
    """Train flow matching model with according to diffusion type.

    Args:
      model: UNet mapping (x,t,train,cond) -> x'
      dataloader: callable which produces an iterator for the minibatches
      data_std: standard deviation of training data for input normalization
      epochs: number of epochs to train
      lr: learning rate
      diffusion: diffusion object (VarianceExploding, VariancePreserving, etc)
      cond_fn: (optional) function cond_fn(x) to condition training on
      num_ema_foldings: number of ema timescales per total number of epochs
      writer: optional summary_writer to log to if not None
      report: optional report function to call if not None
      ckpt: optional clu.checkpoint to save the model. If None, does not save
      seed: random seed for model init and training

    Returns:
      score function (xt,t,cond)->scores (s(xₜ,t):=∇logp(xₜ))
    """
    if key is None:
        key = jax.random.key(rng_seed)
    # initialize model
    key, key_model_init = random.split(key)
    x = next(dataloader())
    params = model.init(
        key_model_init,
        x=x, t=jnp.zeros(x.shape[0]),
        train=False, cond=cond_fn(x),
    )
    log.info('Param count: %(param_count).2f M', dict(param_count=count_params(params['params']) / 1e6))

    @jax.jit
    def loss(params, x_data, key):
        key, key_time = jax.random.split(key)
        t = jax.random.uniform(key_time, shape=(x_data.shape[0], 1, 1))

        key, key_noise = jax.random.split(key)
        x_noise = jax.random.normal(key_noise, x_data.shape)

        if isinstance(cfg.conditional_flow, cs.ConditionalOT):
            xt = (1 - t) * x_noise + t * x_data
        elif isinstance(cfg.conditional_flow, cs.MinibatchOTConditionalOT):
            key, key_plan = jax.random.split(key)
            x_noise, x_data = optimal_transport.OTPlanSamplerJax.sample_plan(
                key_plan,
                x_noise, x_data,
                reg=cfg.conditional_flow.sinkhorn_regularization,
                replace=cfg.conditional_flow.sample_with_replacement,
            )
            xt = (1 - t) * x_noise + t * x_data
        else:
            raise ValueError(f'Unknown conditional flow: {cfg.conditional_flow}')

        velocity_pred = model.apply(params, x=xt, t=t.squeeze((1, 2)), train=True, cond=cond_fn(x_data))
        velocity_target = x_data - x_noise
        return ((velocity_pred - velocity_target)**2).mean()

    tx = optax.adam(learning_rate=cfg.architecture.learning_rate)
    opt_state = tx.init(params)
    ema_ts = cfg.architecture.epochs / num_ema_foldings  # number of ema timescales during training
    ema_params = params
    loss_grad_fn = jax.value_and_grad(loss)

    @jax.jit
    def update_fn(params, ema_params, opt_state, key, data):
        key, key_loss = random.split(key)
        loss_val, grads = loss_grad_fn(params, data, key_loss)

        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        ema_update = lambda p, ema: ema + (p - ema) / ema_ts
        ema_params = jax.tree_map(ema_update, params, ema_params)

        return params, ema_params, opt_state, key, loss_val

    for epoch in range(cfg.architecture.epochs + 1):
        for i, data in enumerate(dataloader()):
            params, ema_params, opt_state, key, loss_val = update_fn(
                params, ema_params, opt_state, key, data
            )
        if epoch % 25 == 0:
            ema_loss = loss(ema_params, data, key)  # pylint: disable=undefined-loop-variable
            if writer is not None:
                metrics = {"loss": loss_val, "ema_loss": ema_loss}
                eval_metrics_cpu = jax.tree_map(np.array, metrics)
                writer.write_scalars(epoch, eval_metrics_cpu)

    model_state = ema_params
    if ckpt is not None:
        ckpt.save(model_state)


def count_params(params):
    """Count the number of parameters in the flax model param dict."""
    if isinstance(params, jax.numpy.ndarray):
        return np.prod(params.shape)
    elif isinstance(params, (dict, FrozenDict)):
        return sum([count_params(v) for v in params.values()])
    else:
        assert False, type(params)
