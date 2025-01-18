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


def heun_sample(key, tmax, velocity, x_shape, nsteps=1000, keep_path=False):
    x_noise = jax.random.normal(key, x_shape)
    timesteps = (.5 + jnp.arange(nsteps)) / nsteps
    x0, xs = samplers.heun_integrate(velocity, x_noise, timesteps)
    return xs if keep_path else x0


def heun_sample_diffusion(key, diffusion, tmax, velocity, x_shape, nsteps=1000, keep_path=False):
    x_noise = jax.random.normal(key, x_shape) * diffusion.sigma(tmax)
    timesteps = (.5 + jnp.arange(nsteps)) / nsteps
    x0, xs = samplers.heun_integrate(velocity, x_noise, timesteps)
    return xs if keep_path else x0


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
        samples = self.sample(key_val, 1., cond, batch.shape, params=self.params_ema)
        return dict(
            val_relative_error=torch.tensor(einops.reduce(utils.relative_error(batch, samples), 'b t ->', 'mean').item()),
        )

    def predict_step(self):
        pass

    def sample(self, key, tmax, cond, x_shape, params=None, keep_path=False, use_score=False):
        if params is None:
            params = self.params_ema

        def velocity(x, t):
            if not hasattr(t, 'shape') or not t.shape:
                t = jnp.ones((x_shape[0], 1, 1)) * t
            return self.velocity(x, t, cond, params)

        if isinstance(self.cfg.model.conditional_flow, cs.ConditionalSDE):
            if use_score:
                def score(x, t):
                    if not hasattr(t, 'shape') or not t.shape:
                        t = jnp.ones((x_shape[0], 1, 1)) * t
                    return self.score(x, t, cond, params)

                return samplers.sde_sample(self.diffusion, score, key, x_shape, nsteps=self.cfg.model.time_step_count_sampling, traj=keep_path)
            else:
                return heun_sample_diffusion(key, self.diffusion, tmax, velocity, x_shape=x_shape, nsteps=self.cfg.model.time_step_count_sampling, keep_path=keep_path)
        else:
            if use_score:
                raise ValueError(
                    f'Writing the score function in terms of the flow matching vector field is only supported when cfg.model.conditional_flow is {cs.ConditionalSDE.__name__}, not {type(self.cfg.model.conditional_flow)}.'
                    'Please set use_score=False.'
                )
            return heun_sample(key, tmax, velocity, x_shape=x_shape, nsteps=self.cfg.model.time_step_count_sampling, keep_path=keep_path)

    @functools.partial(jax.jit, static_argnames=['self'])
    def score(self, x, t, cond, params):
        if not isinstance(self.cfg.model.conditional_flow, cs.ConditionalSDE):
            raise ValueError(
                f'Writing the score function in terms of the flow matching vector field is only supported when cfg.model.conditional_flow is {cs.ConditionalSDE.__name__}, not {self.cfg.model.conditional_flow.__class__.__name__}.'
            )
        if not isinstance(self.cfg.model.conditional_flow.sde_diffusion, cs.SDEVarianceExploding):
            raise ValueError(
                f'Writing the score function in terms of the flow matching vector field is only implemented for when cfg.model.conditional_flow.sde_diffusion is {cs.SDEVarianceExploding.__name__}, not {self.cfg.model.conditional_flow.sde_diffusion.__class__.__name__}.'
            )
        # sde_sample integrates from 1 to 0, so
        # 1. drop the negative sign
        # 2. pass the reversed time to the flow matching model
        return 2 / self.diffusion.g2(t) * self.velocity(x, 1 - t, cond, params)

    @functools.partial(jax.jit, static_argnames=['self', 'train'])
    def velocity(self, x, t, cond, params, train=False):
        if isinstance(self.cfg.model.conditional_flow, cs.ConditionalSDE):
            if self.cfg.model.conditional_flow.finzi_karras_weighting:
                # scaling is equivalent to that in Karras et al. https://arxiv.org/abs/2206.00364
                sigma = self.diffusion.sigma(1 - t)
                # Taos: Karras et al. $c_in$ and $s(t)$ of EDM.
                input_scale = 1 / jnp.sqrt(sigma**2 + self.train_data_std**2)
                cond = cond / self.train_data_std if cond is not None else None
                out = self.model.apply(params, x=x * input_scale, t=t.squeeze((1, 2)), train=train, cond=cond)
                # Taos: Karras et al. the demonimator of $c_out$ of EDM; where is the numerator?
                return out / jnp.sqrt(sigma**2 + self.train_data_std**2)
            else:
                return self.model.apply(params, x=x, t=t.squeeze((1, 2)), train=train, cond=cond)
        else:
            return self.model.apply(params, x=x, t=t.squeeze((1, 2)), train=train, cond=cond)

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
        sigma = self.diffusion.sigma(1 - t)
        xt = x_data + sigma * x_noise
        dsigma = self.diffusion.dsigma(1 - t)
        velocity_target = -dsigma / sigma * (xt - x_data)
        return xt, velocity_target, dsigma

    @functools.partial(jax.jit, static_argnames=['self'])
    def loss(self, key, x_data, cond, params):
        if self.cfg.model.time_samples_uniformly_spaced:
            key, key_time = jax.random.split(key)
            u0 = jax.random.uniform(key_time)
            u = jnp.remainder(u0 + jnp.linspace(0, 1, x_data.shape[0]), 1)
            t = u * (self.diffusion.tmax - self.diffusion.tmin) + self.diffusion.tmin
            t = t[:, None, None]
        else:
            key, key_time = jax.random.split(key)
            t = jax.random.uniform(key_time, shape=(x_data.shape[0], 1, 1))

        key, key_noise = jax.random.split(key)
        x_noise = jax.random.normal(key_noise, x_data.shape)

        if isinstance(self.cfg.model.conditional_flow, cs.ConditionalOT):
            xt, velocity_target = self.conditional_ot(t, x_noise, x_data)
            weighting = 1.
        elif isinstance(self.cfg.model.conditional_flow, cs.MinibatchOTConditionalOT):
            key, key_plan = jax.random.split(key)
            xt, velocity_target = self.minimatch_ot_conditional_ot(key_plan, t, x_noise, x_data)
            weighting = 1.
        elif isinstance(self.cfg.model.conditional_flow, cs.ConditionalSDE):
            if isinstance(self.cfg.model.conditional_flow.sde_diffusion, cs.SDEVarianceExploding):
                xt, velocity_target, dsigma = self.variance_exploding_conditional(t, x_noise, x_data)
                weighting = 1 / dsigma**2
            else:
                raise ValueError(f'Unknown SDE diffusion: {self.cfg.model.conditional_flow.sde_diffusion}')
        else:
            raise ValueError(f'Unknown conditional flow: {self.cfg.model.conditional_flow}')

        velocity_pred = self.velocity(xt, t, cond, params, train=True)
        return ((velocity_pred - velocity_target)**2 * weighting).mean()

    @functools.partial(jax.jit, static_argnames=['self'])
    def step(self, key, batch, cond, params, params_ema, opt_state):
        loss, grads = self.loss_and_grad(key, batch, cond, params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        ema_update = lambda p, ema: ema + (p - ema) / self.ema_ts
        params_ema = jax.tree.map(ema_update, params, params_ema)
        return loss, params, params_ema, opt_state
