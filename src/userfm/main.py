import functools
import logging
import pprint
import sys

import hydra
from omegaconf import OmegaConf
import tensorflow as tf
import einops
import torch
import torch.utils.data.dataloader
import numpy as np
import jax
import jax.numpy as jnp
import lightning.pytorch as pl
import optax

from userdiffusion import samplers, unet
from userfm import callbacks, cs, datasets, diffusion, flow_matching, sde_diffusion, utils


log = logging.getLogger(__name__)


@functools.partial(jax.jit, static_argnames='integrate')
def log_prediction_metric(qs, times, integrate):
    """
    Log geometric mean of rollout relative error computed over a trajectory.
    """
    trajectory = qs
    trajectory_groud_truth = integrate(trajectory[0], times)
    # Taos: todo: why does relative_error return a vector?
    return jnp.log(
        utils.relative_error(trajectory, trajectory_groud_truth)[1:len(times)//2]
    ).mean()


@functools.partial(jax.jit, static_argnames='integrate')
def pmetric(qs, times, integrate):
    """
    Geometric mean of rollout relative error, also taken over the batch.
    """
    log_metric = jax.vmap(
        functools.partial(log_prediction_metric, times=times, integrate=integrate)
    )(qs)
    std_err = jnp.exp(log_metric.std() / jnp.sqrt(log_metric.shape[0]))
    return jnp.exp(log_metric.mean()), std_err


def condition_on_initial_time_steps(z, time_step_count):
    if time_step_count > 0:
        return z[:, :time_step_count]
    return None


class Trainer(pl.LightningModule):
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

        self.diffusion = sde_diffusion.get_sde_diffusion(self.cfg.model.sde_diffusion)
        self.ema_ts = self.cfg.model.architecture.epochs / 5  # num_ema_foldings  # number of ema timescales during training

    def setup(self, stage):
        if stage == 'fit':
            self.key, key_train = jax.random.split(self.key)
            self.params = self.model_init(key_train, self.x_shape, self.cond_fn, self.model)
            self.ema_params = self.params
            # self.optimizer, self.opt_state = self.configure_optimizers(self.params)
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
        self.key, key_train = jax.random.split(self.key)
        loss, self.params, self.ema_params, self.opt_state = Trainer.step(
            key_train, batch, self.train_data_std, self.cond_fn(batch),
            self.model, self.params, self.ema_params, self.ema_ts,
            self.diffusion,
            self.optimizer, self.opt_state,
        )

        self.optimizers()  # increment global step for logging and checkpointing
        return dict(loss=torch.tensor(loss.item()))

    def val_dataloader(self):
        # from pytorch_lightning.utilities import CombinedLoader
        return self.dataloaders['val']

    def validation_step(self, batch, batch_idx):
        self.key, key_val = jax.random.split(self.key)
        cond = self.cond_fn(batch)
        def score(x, t):
            if not hasattr(t, "shape") or not t.shape:
                t = jnp.ones(x.shape[0]) * t
            return Trainer.score(x, t, self.train_data_std, cond, self.model, self.params, self.diffusion, train=False)

        samples = samplers.sde_sample(self.diffusion, score, key_val, x_shape=batch.shape, nsteps=1_000)
        return dict(
            val_relative_error=torch.tensor(einops.reduce(utils.relative_error(batch, samples), 'b t ->', 'mean').item()),
        )

    def predict_step(self):
        pass

    @staticmethod
    @functools.partial(jax.jit, static_argnames=['model', 'diffusion', 'train'])
    def score(x, t, train_data_std, cond, model, params, diffusion, train=True):
        """Score function with appropriate input and output scaling."""
        # scaling is equivalent to that in Karras et al. https://arxiv.org/abs/2206.00364
        sigma, scale = utils.unsqueeze_like(x, diffusion.sigma(t), diffusion.scale(t))
        # Taos: Karras et al. $c_in$ and $s(t)$ of EDM.
        input_scale = 1 / jnp.sqrt(sigma**2 + (scale * train_data_std) ** 2)
        cond = cond / train_data_std if cond is not None else None
        out = model.apply(params, x=x * input_scale, t=t, train=train, cond=cond)
        # Taos: Karras et al. the demonimator of $c_out$ of EDM; where is the numerator?
        return out / jnp.sqrt(sigma**2 + scale**2 * train_data_std**2)

    @staticmethod
    @functools.partial(jax.value_and_grad, argnums=5)
    @functools.partial(jax.jit, static_argnames=['model', 'diffusion'])
    def loss(key, x, train_data_std, cond, model, params, diffusion):
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
        # Taos: this appears to be using the weighting from Eqn.(1) used for discrete noise levels.
        weighting = utils.unsqueeze_like(x, diffusion.sigma(t) ** 2)
        error = Trainer.score(x, t, train_data_std, cond, model, params, diffusion, train=True) - target_score
        return jnp.mean((diffusion.covsqrt.inverse(error) ** 2) * weighting)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=['model', 'diffusion', 'optimizer'])
    def step(key, batch, train_data_std, cond, model, params, ema_params, ema_ts, diffusion, optimizer, opt_state):
        loss, grads = Trainer.loss(key, batch, train_data_std, cond, model, params, diffusion)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        ema_update = lambda p, ema: ema + (p - ema) / ema_ts
        ema_params = jax.tree.map(ema_update, params, ema_params)
        return loss, params, ema_params, opt_state


@hydra.main(**utils.HYDRA_INIT)
def main(cfg):
    engine = cs.get_engine()
    cs.create_all(engine)
    with cs.orm.Session(engine, expire_on_commit=False) as db:
        cfg = cs.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        pprint.pp(cfg)
        log.info(f'Outputs will be saved to: {cfg.run_dir}')

        # Hide GPUs from Tensorflow to prevent it from reserving memory,
        # and making it unavailable to JAX.
        tf.config.experimental.set_visible_devices([], 'GPU')

        log.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
        log.info('JAX devices: %r', jax.devices())

        key = jax.random.key(cfg.rng_seed)
        key, key_dataset = jax.random.split(key)
        ds = datasets.get_dataset(cfg.dataset, key=key_dataset)
        splits = datasets.split_dataset(cfg.dataset, ds)
        if splits['train'].shape[1] != 60:
            log.warn(
                'Finzi et al., 2023, trim the trajectories to include only first 60 time steps after the "burn-in" time steps, but these trajectories have %(time_steps)d time steps.'
                'Consider setting dataset.time_step_count equal to dataset.time_step_count_drop_first + 60.',
                dict(time_steps=splits['train'].shape[1])
            )
        dataloaders = {}
        for n, s in splits.items():
            dataloaders[n] = (
                torch.utils.data.dataloader.DataLoader(
                    list(tf.data.Dataset.from_tensor_slices(s).batch(cfg.dataset.batch_size).as_numpy_iterator()),
                    batch_size=1,
                    collate_fn=lambda x: x[0],
                    # batch_size=cfg.dataset.batch_size,
                    # collate_fn=lambda x: jnp.stack(x)
                )
                # tf.data.Dataset.from_tensor_slices(s)
                # .shuffle(len(s))
                # .batch(cfg.dataset.batch_size)
                # .as_numpy_iterator
            )

        cfg_unet = unet.unet_64_config(
            splits['train'].shape[2],
            base_channels=cfg.model.architecture.base_channel_count,
            attention=cfg.model.architecture.attention,
        )
        model = unet.UNet(cfg_unet)

        train_data_std = splits['train'].std()
        log.info('Training set standard deviation: %(data_std).7f', dict(data_std=train_data_std))

        cond_fn = functools.partial(condition_on_initial_time_steps, time_step_count=cfg.dataset.time_step_count_conditioning)
        key, key_trainer = jax.random.split(key)
        if isinstance(cfg.model, cs.ModelDiffusion):
            trainer = Trainer(cfg, key_trainer, dataloaders, train_data_std, cond_fn, model)
        elif isinstance(cfg.model, cs.ModelFlowMatching):
            trainer = flow_matching.Trainer(cfg)
        else:
            raise ValueError(f'Unknown model: {cfg.model}')

        logger = pl.loggers.TensorBoardLogger(cfg.run_dir, version=0)

        pl_trainer = pl.Trainer(
            max_epochs=cfg.model.architecture.epochs,
            logger=logger,
            precision=32,
            callbacks=[
                callbacks.ModelCheckpoint(
                    dirpath=cfg.run_dir,
                    filename='{epoch}',
                    save_last=True,
                    save_top_k=2,
                    monitor='val_relative_error',
                    save_on_train_epoch_end=False,
                ),
                callbacks.LogStats(),
            ],
            check_val_every_n_epoch=25,
            deterministic=True,
        )

        pl_trainer.fit(trainer)


def get_run_dir(hydra_init=utils.HYDRA_INIT, commit=True):
    with hydra.initialize(version_base=hydra_init['version_base'], config_path=hydra_init['config_path']):
        first_override = None
        overrides = []
        for i, a in enumerate(sys.argv):
            if '=' in a:
                overrides.append(a)
                first_override = i
        cfg = hydra.compose(hydra_init['config_name'], overrides=overrides)
        engine = cs.get_engine()
        cs.create_all(engine)
        with cs.orm.Session(engine, expire_on_commit=False) as db:
            cfg = cs.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
            if commit:
                db.commit()
            return first_override, str(cfg.run_dir)


if __name__ == '__main__':
    first_override, run_dir = get_run_dir()
    run_dir_override = f'hydra.run.dir={run_dir}'
    if first_override is None:
        sys.argv.append(run_dir_override)
    else:
        sys.argv.insert(first_override, run_dir_override)
    main()
