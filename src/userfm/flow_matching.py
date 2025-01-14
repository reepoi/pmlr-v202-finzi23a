import functools
import logging

import einops
import jax
import jax.numpy as jnp
import torch
import lightning.pytorch as pl
import optax

from userdiffusion import samplers
from userfm import cs, optimal_transport, sde_diffusion, utils


log = logging.getLogger(__file__)


def heun_sample(key, tmax, velocity, x_shape, nsteps=1000, traj=False):
  x_noise = jax.random.normal(key, x_shape)
  timesteps = (.5 + jnp.arange(nsteps)) / nsteps
  x0, xs = samplers.heun_integrate(velocity, x_noise, timesteps)
  return xs if traj else x0


class JaxLightning(pl.LightningModule):
    def __init__(self, cfg, key, dataloaders, train_data_std, cond_fn, model):
        super().__init__()
        self.automatic_optimization = False

        self.cfg = cfg
        self.key = key
        self.dataloaders = dataloaders
        self.train_data_std = train_data_std
        self.x_shape = next(iter(dataloaders['train'])).shape
        self.cond_fn = cond_fn
        self.model = model
        if isinstance(self.cfg.model.conditional_flow, cs.ConditionalSDE):
            self.diffusion = sde_diffusion.get_sde_diffusion(self.cfg.model.conditional_flow.sde_diffusion)

        self.ema_ts = self.cfg.model.architecture.epochs / self.cfg.model.architecture.ema_folding_count

        self.loss_and_grad = jax.value_and_grad(self.loss, argnums=3)

    def __hash__(self):
        return hash(id(self))

    def setup(self, stage):
        if stage == 'fit':
            self.key, key_train = jax.random.split(self.key)
            self.params = self.model_init(key_train, self.x_shape, self.cond_fn, self.model)
            self.params_ema = self.params
        elif stage == 'val':
            pass
        else:
            raise ValueError(f'Unknown stage: {stage}')

    def model_init(self, key, x_shape, cond_fn, model):
        x = jnp.ones(x_shape)
        t = jnp.ones(x_shape[0])
        cond = cond_fn(x)
        params = model.init(key, x=x, t=t, train=False, cond=cond)
        return params

    def configure_optimizers(self):
        self.optimizer = optax.adam(learning_rate=self.cfg.model.architecture.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    def train_dataloader(self):
        return self.dataloaders['train']

    def training_step(self, batch, batch_idx):
        cond = self.cond_fn(batch)
        self.key, key_train = jax.random.split(self.key)
        loss, self.params, self.params_ema, self.opt_state = self.step(
            key_train, batch, cond,
            self.params, self.params_ema,
            self.opt_state,
        )
        # use same key to ensure identical sampling
        loss_ema = self.loss(key_train, batch, cond, self.params_ema)
        self.optimizers().step()  # increment global step for logging and checkpointing
        return dict(
            loss=torch.tensor(loss.item()),
            loss_ema=torch.tensor(loss_ema.item()),
        )

    def val_dataloader(self):
        # from pytorch_lightning.utilities import CombinedLoader
        return self.dataloaders['val']

    def validation_step(self, batch, batch_idx):
        self.key, key_val = jax.random.split(self.key)
        if self.cfg.dataset.time_step_count_conditioning == 0:
            return dict(
                val_relative_error=torch.tensor(0.),
            )
        cond = self.cond_fn(batch)
        def velocity(x, t):
            if not hasattr(t, 'shape') or not t.shape:
                t = jnp.ones(x.shape[0]) * t
            if isinstance(self.cfg.model.conditional_flow, cs.ConditionalSDE):
                if isinstance(self.cfg.model.conditional_flow.sde_diffusion, cs.SDEVarianceExploding):
                    return -self.velocity(x, t, cond, self.params_ema)
                else:
                    raise ValueError(f'Unknown SDE diffusion: {self.cfg.model.conditional_flow.sde_diffusion}')
            else:
                return self.velocity(x, t, cond, self.params_ema)

        samples = heun_sample(key_val, 1., velocity, x_shape=batch.shape, nsteps=self.cfg.model.ode_time_steps)
        return dict(
            val_relative_error=torch.tensor(einops.reduce(utils.relative_error(batch, samples), 'b t ->', 'mean').item()),
        )

    def predict_step(self):
        pass

    @functools.partial(jax.jit, static_argnames=['self', 'train'])
    def velocity(self, x, t, cond, params, train=False):
        return self.model.apply(params, x=x, t=t, train=train, cond=cond)

    @functools.partial(jax.jit, static_argnames=['self'])
    def conditional_ot(self, t, x_noise, x_data):
        xt = (1 - t) * x_noise + t * x_data
        velocity_target = x_data - x_noise
        return xt, velocity_target

    @functools.partial(jax.jit, static_argnames=['self'])
    def minimatch_ot_conditional_ot(self, key, t, x_noise, x_data):
        x_noise, x_data = optimal_transport.OTPlanSamplerJax.sample_plan(
            key,
            x_noise, x_data,
            reg=self.cfg.model.conditional_flow.sinkhorn_regularization,
            replace=self.cfg.model.conditional_flow.sample_with_replacement,
        )
        return self.conditional_ot(t, x_noise, x_data)

    @functools.partial(jax.jit, static_argnames=['self'])
    def variance_exploding_conditional(self, t, x_noise, x_data):
        sigma, minus_dsigma = jax.vmap(jax.value_and_grad(self.diffusion.sigma))(
            (1 - t).squeeze((1, 2))
        )
        sigma, minus_dsigma = sigma[:, None, None], minus_dsigma[:, None, None]
        # sigma = self.diffusion.sigma(1 - t)
        # minus_dsigma = jax.vmap(self.diffusion.sigma)((1 - t).squeeze((1, 2)))[:, None, None]
        xt = x_data + sigma * x_noise
        # velocity_target = minus_dsigma * x_noise
        velocity_target = minus_dsigma / sigma * (xt - x_data)
        # sigma = self.diffusion.sigma(1 - t)
        # xt = x_data + sigma * x_noise
        # minus_dsigma = jax.vmap(self.diffusion.sigma)((1 - t).squeeze((1, 2)))[:, None, None]
        # velocity_target = minus_dsigma / sigma * (xt - x_data)
        return xt, velocity_target

    @functools.partial(jax.jit, static_argnames=['self'])
    def loss(self, key, x_data, cond, params):
        if isinstance(self.cfg.model.conditional_flow, cs.ConditionalOT):
            key, key_time = jax.random.split(key)
            t = jax.random.uniform(key_time, shape=(x_data.shape[0], 1, 1))

            key, key_noise = jax.random.split(key)
            x_noise = jax.random.normal(key_noise, x_data.shape)

            xt, velocity_target = self.conditional_ot(t, x_noise, x_data)
        elif isinstance(self.cfg.model.conditional_flow, cs.MinibatchOTConditionalOT):
            key, key_time = jax.random.split(key)
            t = jax.random.uniform(key_time, shape=(x_data.shape[0], 1, 1))

            key, key_noise = jax.random.split(key)
            x_noise = jax.random.normal(key_noise, x_data.shape)

            key, key_plan = jax.random.split(key)
            xt, velocity_target = self.minimatch_ot_conditional_ot(key_plan, t, x_noise, x_data)
        elif isinstance(self.cfg.model.conditional_flow, cs.ConditionalSDE):
            # key, key_time = jax.random.split(key)
            # u0 = jax.random.uniform(key_time)
            # u = jnp.remainder(u0 + jnp.linspace(0, 1, x_data.shape[0]), 1)
            # t = u * (self.diffusion.tmax - self.diffusion.tmin) + self.diffusion.tmin
            # t = t[:, None, None]
            key, key_time = jax.random.split(key)
            t = jax.random.uniform(key_time, shape=(x_data.shape[0], 1, 1))

            key, key_noise = jax.random.split(key)
            x_noise = jax.random.normal(key_noise, x_data.shape)

            # weighting = self.diffusion.sigma(1 - t)**2
            if isinstance(self.cfg.model.conditional_flow.sde_diffusion, cs.SDEVarianceExploding):
                xt, velocity_target = self.variance_exploding_conditional(t, x_noise, x_data)
            else:
                raise ValueError(f'Unknown SDE diffusion: {self.cfg.model.conditional_flow.sde_diffusion}')
        else:
            raise ValueError(f'Unknown conditional flow: {self.cfg.model.conditional_flow}')

        velocity_pred = self.velocity(xt, t.squeeze((1, 2)), cond, params, train=True)
        return ((velocity_pred - velocity_target)**2).mean()

    @functools.partial(jax.jit, static_argnames=['self'])
    def step(self, key, batch, cond, params, params_ema, opt_state):
        loss, grads = self.loss_and_grad(key, batch, cond, params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        ema_update = lambda p, ema: ema + (p - ema) / self.ema_ts
        params_ema = jax.tree.map(ema_update, params, params_ema)
        return loss, params, params_ema, opt_state
