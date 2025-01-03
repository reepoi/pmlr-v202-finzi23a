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

"""Core diffusion model implementation, noise schedule, type, and training."""

import time
import logging
from typing import Any, Callable, Iterator, List, Optional, Sequence, Union

from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
import jax
from jax import grad
from jax import jit
from jax import random
import jax.numpy as jnp
import numpy as np
import optax

from userfm import sde_diffusion, utils

log = logging.getLogger(__file__)

Scorefn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
PRNGKey = jnp.ndarray
TimeType = Union[float, jnp.ndarray]
ArrayShape = Sequence[int]
ParamType = Any


def nonefn(x):  # pylint: disable=unused-argument
    return None


def train_diffusion(
    cfg,
    model,
    diffusion,
    dataloader,
    data_std,
    cond_fn=nonefn,  # function: array -> array or None
    num_ema_foldings=5,
    writer=None,
    report=None,
    ckpt=None,
    key=None,
    rng_seed=None,  # to avoid initing jax
):
    """Train diffusion model with score matching according to diffusion type.

    Minimizes score matching MSE loss between the model scores s(xₜ,t)
    and the data scores ∇log p(xₜ|x₀) over noised datapoints xₜ, with t sampled
    uniformly from 0 to 1, and x sampled from the training distribution.
    Produces score function s(xₜ,t) ≈ ∇log p(xₜ) which can be used for sampling.

    Loss = 𝔼[|s(xₜ,t) − ∇log p(xₜ|x₀)|²σₜ²]

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
    if key is None:
        key = jax.random.key(rng_seed)
    key, init_seed = random.split(key)
    params = model.init(init_seed, x=x, t=t, train=False, cond=cond_fn(x))
    log.info(f"{count_params(params['params'])/1e6:.2f}M Params")  # pylint: disable=logging-fstring-interpolation

    def score(params, x, t, train=True, cond=None):
        """Score function with appropriate input and output scaling."""
        # scaling is equivalent to that in Karras et al. https://arxiv.org/abs/2206.00364
        sigma, scale = utils.unsqueeze_like(x, diffusion.sigma(t), diffusion.scale(t))
        # Taos: Karras et al. $c_in$ and $s(t)$ of EDM.
        input_scale = 1 / jnp.sqrt(sigma**2 + (scale * data_std) ** 2)
        cond = cond / data_std if cond is not None else None
        out = model.apply(params, x=x * input_scale, t=t, train=train, cond=cond)
        # Taos: Karras et al. the demonimator of $c_out$ of EDM; where is the numerator?
        return out / jnp.sqrt(sigma**2 + scale**2 * data_std**2)

    def loss(params, x, key):
        """Score matching MSE loss from Yang's Score-SDE paper."""
        # Use lowvar grid time sampling from https://arxiv.org/pdf/2107.00630.pdf
        # Appendix I
        key, key_time = jax.random.split(key)
        u0 = jax.random.uniform(key_time)
        u = jnp.remainder(u0 + jnp.linspace(0, 1, x.shape[0]), 1)
        t = u * (diffusion.tmax - diffusion.tmin) + diffusion.tmin

        key, key_noise = jax.random.split(key)
        xt = diffusion.noise_input(x, t, key_noise)
        target_score = diffusion.noise_score(xt, x, t)
        # weighting from Yang Song's https://arxiv.org/abs/2011.13456
        # Taos: this appears to be using the weighting from Eqn.(1) used for dicrete noise levels.
        weighting = utils.unsqueeze_like(x, diffusion.sigma(t) ** 2)
        error = score(params, xt, t, cond=cond_fn(x)) - target_score
        return jnp.mean((diffusion.covsqrt.inverse(error) ** 2) * weighting)

    tx = optax.adam(learning_rate=cfg.architecture.learning_rate)
    opt_state = tx.init(params)
    ema_ts = cfg.architecture.epochs / num_ema_foldings  # number of ema timescales during training
    ema_params = params
    jloss = jit(loss)
    loss_grad_fn = jax.value_and_grad(loss)

    @jit
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
            ema_loss = jloss(ema_params, data, key)  # pylint: disable=undefined-loop-variable
            if writer is not None:
                metrics = {"loss": loss_val, "ema_loss": ema_loss}
                eval_metrics_cpu = jax.tree_map(np.array, metrics)
                writer.write_scalars(epoch, eval_metrics_cpu)

    model_state = ema_params
    if ckpt is not None:
        ckpt.save(model_state)

    @jit
    def score_out(x, t, cond=None):
        """Trained score function s(xₜ,t):=∇logp(xₜ)."""
        if not hasattr(t, "shape") or not t.shape:
            t = jnp.ones(x.shape[0]) * t
        return score(ema_params, x, t, train=False, cond=cond)

    return score_out


def count_params(params):
    """Count the number of parameters in the flax model param dict."""
    if isinstance(params, jax.numpy.ndarray):
        return np.prod(params.shape)
    elif isinstance(params, (dict, FrozenDict)):
        return sum([count_params(v) for v in params.values()])
    else:
        assert False, type(params)
