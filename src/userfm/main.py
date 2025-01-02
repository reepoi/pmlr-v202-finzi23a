import functools
import logging
import pickle
import sys

import hydra
from omegaconf import OmegaConf
import tensorflow as tf
import jax
import jax.numpy as jnp
import clu.checkpoint
import clu.platform

from userdiffusion import unet, samplers

import cs
import datasets
import flow_matching


log = logging.getLogger(__name__)


def train(cfg):
    ds = datasets.get_dataset(cfg.dataset, rng_seed=cfg.rng_seed)
    trajectories = ds.Zs[cfg.dataset.batch_size:]
    if trajectories.shape[1] != 60:
        log.warn(
            'Finzi et al., 2023, trim the trajectories to include only first 60 time steps after the "burn-in" time steps, but these trajectories have %(time_steps)d time steps.'
            'Consider setting dataset.time_step_count equal to dataset.time_step_count_drop_first + 60.',
            dict(time_steps=trajectories.shape[1])
        )
    test_x = ds.Zs[:cfg.dataset.batch_size]
    data_std = trajectories.std()
    T_long = ds.T_long

    dataset = tf.data.Dataset.from_tensor_slices(trajectories)
    dl = dataset.shuffle(len(dataset)).batch(cfg.dataset.batch_size).as_numpy_iterator

    cfg_unet = unet.unet_64_config(
        test_x.shape[-1],
        base_channels=32,  # Taos: todo: parametrize this in config,
        attention=False,  # Taos: todo: parametrize this in config,
    )
    model = unet.UNet(cfg_unet)
    cond_fn = lambda z: (z[:3, :3] if False else None)  # Taos: todo: parametrize this in config,

    ckpt = clu.checkpoint.MultihostCheckpoint(str(cfg.run_dir/'checkpoints'), {}, max_to_keep=2)

    velocity = flow_matching.train_flow_matching(
        model, dl, data_std,
        epochs=cfg.model.epochs,
        lr=cfg.model.learning_rate,
        ckpt=ckpt,
        cond_fn=cond_fn,
    )

    @jax.jit
    def relative_error(x, y, axis=-1):
        """
        Compute |x - y|/(|x| + |y|) with L1 norm over the axis `axis`.
        """
        return (
            jnp.abs(x - y).sum(axis)
            / (jnp.abs(x).sum(axis) + jnp.abs(y).sum(axis))
        )

    # Taos: todo: parametrize kstart in config
    @jax.jit
    def log_prediction_metric(qs, kstart=3):
        """
        Log geometric mean of rollout relative error computed over a trajectory.
        """
        trajectory = qs[kstart:]
        times = T_long[kstart:]
        trajectory_groud_truth = ds.integrate(trajectory[0], times)
        # Taos: todo: why does relative_error return a vector?
        return jnp.log(
            relative_error(trajectory, trajectory_groud_truth)[1:len(times)//2]
        ).mean()

    @jax.jit
    def pmetric(qs):
        """
        Geometric mean of rollout relative error, also taken over the batch.
        """
        log_metric = jax.vmap(log_prediction_metric)(qs)
        std_err = jnp.exp(log_metric.std() / jnp.sqrt(log_metric.shape[0]))
        return jnp.exp(log_metric.mean()), std_err

    eval_velocity = functools.partial(velocity, cond_fn(test_x))
    key = jax.random.PRNGKey(cfg.rng_seed)
    nll = samplers.compute_nll(..., eval_velocity, key, test_x)
    stochastic_samples = samplers.sde_sample(
        ..., eval_velocity, key, test_x.shape,
        nsteps=1_000, traj=False,
    )
    err = pmetric(stochastic_samples)[0]

    log.info('NLL: %(nll).3f, Err: %(err).3f', dict(nll=nll, err=err))


HYDRA_INIT = dict(version_base=None, config_path='../../conf', config_name='config')


@hydra.main(**HYDRA_INIT)
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    engine = cs.get_engine()
    cs.create_all(engine)
    with cs.orm.Session(engine, expire_on_commit=False) as db:
        cfg = cs.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        log.info(f'Outputs will be saved to: {cfg.run_dir}')

        # Hide GPUs from Tensorflow to prevent it from reserving memory,
        # and making it unavailable to JAX.
        tf.config.experimental.set_visible_devices([], 'GPU')

        log.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
        log.info('JAX devices: %r', jax.devices())

        train(cfg)


def get_run_dir(hydra_init=HYDRA_INIT, commit=True):
    with hydra.initialize(version_base=hydra_init['version_base'], config_path=hydra_init['config_path']):
        cfg = hydra.compose(hydra_init['config_name'], overrides=sys.argv[1:])
        engine = cs.get_engine()
        cs.create_all(engine)
        with cs.orm.Session(engine, expire_on_commit=False) as db:
            cfg = cs.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
            if commit:
                db.commit()
            return str(cfg.run_dir)

if __name__ == '__main__':
    run_dir = get_run_dir()
    sys.argv.append(f'hydra.run.dir={run_dir}')
    main()
