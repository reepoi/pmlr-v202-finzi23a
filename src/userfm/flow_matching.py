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

import time
import logging
from typing import Any, Callable, Sequence, Union

from flax.core.frozen_dict import FrozenDict
import jax
from jax import grad
from jax import jit
from jax import random
import jax.numpy as jnp
import numpy as np
import optax

from userfm import sde_diffusion, utils

Scorefn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
PRNGKey = jnp.ndarray
TimeType = Union[float, jnp.ndarray]
ArrayShape = Sequence[int]
ParamType = Any


log = logging.getLogger(__file__)


def nonefn(x):  # pylint: disable=unused-argument
    return None


def train_flow_matching(
    model,
    dataloader,
    data_std,
    epochs=100,
    lr=1e-3,
    cond_fn=lambda _: None,  # function: array -> array or None
    num_ema_foldings=5,
    writer=None,
    report=None,
    ckpt=None,
    seed=None,  # to avoid initing jax
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
    # initialize model
    x = next(dataloader())
    t = np.random.rand(x.shape[0])
    key = random.PRNGKey(42) if seed is None else seed
    key, init_seed = random.split(key)
    params = model.init(init_seed, x=x, t=t, train=False, cond=cond_fn(x))
    log.info('Param count: %(param_count).2f M', dict(param_count=count_params(params['params']) / 1e6))

    def velocity(params, x, t, train=True, cond=None):
        cond = cond / data_std if cond is not None else None
        out = model.apply(params, x=x, t=t, train=train, cond=cond)
        return out

    def loss(params, x_data, key):
        key1, key2 = jax.random.split(key)
        t = jax.random.uniform(key1, shape=(x_data.shape[0], 1, 1))

        x_noise = jax.random.normal(key2, x_data.shape)
        target_velocity = x_data - (1 - 1e-3) * x_noise
        xt = (1 - (1 - 1e-3) * t) * x_noise + t * x_data
        error = velocity(params, xt, t.squeeze((1, 2)), cond=cond_fn(x_data)) - target_velocity
        return jnp.mean(error**2)

    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)
    ema_ts = epochs / num_ema_foldings  # number of ema timescales during training
    ema_params = params
    jloss = jit(loss)
    loss_grad_fn = jax.value_and_grad(loss)

    @jit
    def update_fn(params, ema_params, opt_state, key, data):
        loss_val, grads = loss_grad_fn(params, data, key)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        key, _ = random.split(key)
        ema_update = lambda p, ema: ema + (p - ema) / ema_ts
        ema_params = jax.tree_map(ema_update, params, ema_params)
        return params, ema_params, opt_state, key, loss_val

    for epoch in range(epochs + 1):
        for i, data in enumerate(dataloader()):
            params, ema_params, opt_state, key, loss_val = update_fn(
                params, ema_params, opt_state, key, data
            )
        if epoch % 25 == 0:
            ema_loss = jloss(ema_params, data, key)  # pylint: disable=undefined-loop-variable
            log.info('Epoch %(epoch)d, Val Loss %(loss_val).3f, Ema Loss: %(ema_loss).3f', dict(epoch=epoch, loss_val=loss_val, ema_loss=ema_loss))
            if writer is not None:
                metrics = {"loss": loss_val, "ema_loss": ema_loss}
                eval_metrics_cpu = jax.tree_map(np.array, metrics)
                writer.write_scalars(epoch, eval_metrics_cpu)
                report(epoch, time.time())

    model_state = ema_params
    if ckpt is not None:
        ckpt.save(model_state)

    @jit
    def score_out(x, t, cond=None):
        """Trained score function s(xₜ,t):=∇logp(xₜ)."""
        if not hasattr(t, "shape") or not t.shape:
            t = jnp.ones(x.shape[0]) * t
        return velocity(ema_params, x, t, train=False, cond=cond)

    return score_out


def count_params(params):
    """Count the number of parameters in the flax model param dict."""
    if isinstance(params, jax.numpy.ndarray):
        return np.prod(params.shape)
    elif isinstance(params, (dict, FrozenDict)):
        return sum([count_params(v) for v in params.values()])
    else:
        assert False, type(params)
