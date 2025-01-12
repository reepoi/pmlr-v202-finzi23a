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
            dataloaders[n] = torch.utils.data.dataloader.DataLoader(
                list(tf.data.Dataset.from_tensor_slices(s).batch(cfg.dataset.batch_size).as_numpy_iterator()),
                batch_size=1,
                collate_fn=lambda x: x[0],
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
            jax_lightning = diffusion.JaxLightning(cfg, key_trainer, dataloaders, train_data_std, cond_fn, model)
        elif isinstance(cfg.model, cs.ModelFlowMatching):
            jax_lightning = flow_matching.JaxLightning(cfg, key_trainer, dataloaders, train_data_std, cond_fn, model)
        else:
            raise ValueError(f'Unknown model: {cfg.model}')

        logger = pl.loggers.TensorBoardLogger(cfg.run_dir, name='', version='tb_logs')

        pl_trainer = pl.Trainer(
            max_epochs=cfg.model.architecture.epochs,
            logger=logger,
            precision=32,
            callbacks=[
                callbacks.ModelCheckpoint(
                    dirpath=cfg.run_dir,
                    filename='{epoch}',
                    save_top_k=2,
                    monitor='val_relative_error',
                    save_on_train_epoch_end=False,
                    enable_version_counter=False,
                ),
                callbacks.LogStats(),
            ],
            log_every_n_steps=1,
            check_val_every_n_epoch=cfg.check_val_every_n_epoch,
            deterministic=True,
        )

        pl_trainer.fit(jax_lightning)


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
