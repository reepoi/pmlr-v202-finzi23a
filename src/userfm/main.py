import functools
import logging
import sys

import hydra
from omegaconf import OmegaConf
import tensorflow as tf
import jax
import jax.numpy as jnp
import clu.checkpoint
import clu.platform

from userdiffusion import unet, samplers
from userfm import cs, datasets, diffusion, flow_matching, sde_diffusion


log = logging.getLogger(__name__)


@jax.jit
def relative_error(x, y, axis=-1):
    """
    Compute |x - y|/(|x| + |y|) with L1 norm over the axis `axis`.
    """
    return (
        jnp.abs(x - y).sum(axis)
        / (jnp.abs(x).sum(axis) + jnp.abs(y).sum(axis))
    )


@functools.partial(jax.jit, static_argnames='integrate')
def log_prediction_metric(qs, times, integrate):
    """
    Log geometric mean of rollout relative error computed over a trajectory.
    """
    trajectory = qs
    trajectory_groud_truth = integrate(trajectory[0], times)
    # Taos: todo: why does relative_error return a vector?
    return jnp.log(
        relative_error(trajectory, trajectory_groud_truth)[1:len(times)//2]
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


def train_diffusion(cfg):
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
    log.info('Training set standard deviation: %(data_std).7f', dict(data_std=data_std))
    T_long = ds.T_long

    dataset = tf.data.Dataset.from_tensor_slices(trajectories)
    dl = dataset.shuffle(len(dataset)).batch(cfg.dataset.batch_size).as_numpy_iterator

    cfg_unet = unet.unet_64_config(
        test_x.shape[-1],
        base_channels=cfg.model.architecture.base_channel_count,
        attention=cfg.model.architecture.attention,
    )
    model = unet.UNet(cfg_unet)

    ckpt = clu.checkpoint.MultihostCheckpoint(str(cfg.run_dir/'checkpoints'), {}, max_to_keep=2)

    difftype = sde_diffusion.get_sde_diffusion(cfg.model.sde_diffusion)
    score_fn = diffusion.train_diffusion(
        model, difftype, dl, data_std,
        epochs=cfg.model.architecture.epochs,
        lr=cfg.model.architecture.learning_rate,
        ckpt=ckpt,
    )

    eval_scorefn = functools.partial(score_fn, cond=None)
    key = jax.random.PRNGKey(cfg.rng_seed)
    nll = samplers.compute_nll(difftype, eval_scorefn, key, test_x).mean()
    stochastic_samples = samplers.sde_sample(
        difftype, eval_scorefn, key, test_x.shape,
        nsteps=1000, traj=False
    )
    kstart = 3
    err = pmetric(stochastic_samples[:, kstart:], T_long[kstart:], ds.integrate)[0]

    log.info('NLL: %(nll).3f, Err: %(err).3f', dict(nll=nll, err=err))


def train_flow_matching(cfg):
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
    log.info('Training set standard deviation: %(data_std).7f', dict(data_std=data_std))
    T_long = ds.T_long

    dataset = tf.data.Dataset.from_tensor_slices(trajectories)
    dl = dataset.shuffle(len(dataset)).batch(cfg.dataset.batch_size).as_numpy_iterator

    cfg_unet = unet.unet_64_config(
        test_x.shape[-1],
        base_channels=cfg.model.architecture.base_channel_count,
        attention=cfg.model.architecture.attention,
    )
    model = unet.UNet(cfg_unet)

    ckpt = clu.checkpoint.MultihostCheckpoint(str(cfg.run_dir/'checkpoints'), {}, max_to_keep=2)

    velocity = flow_matching.train_flow_matching(
        model, dl, data_std,
        epochs=cfg.model.architecture.epochs,
        lr=cfg.model.architecture.learning_rate,
        ckpt=ckpt,
    )

    eval_velocity = functools.partial(velocity, cond=None)
    key = jax.random.PRNGKey(cfg.rng_seed)
    cfg_sde_diffusion = OmegaConf.create(dict(
        time_min=1e-3, time_max=1.,
        sigma_min = 1e-3, sigma_max=300.,
    ))
    difftype = sde_diffusion.VarianceExploding(cfg_sde_diffusion)
    nll = samplers.compute_nll(difftype, eval_velocity, key, test_x).mean()
    stochastic_samples = samplers.sde_sample(
        difftype, eval_velocity, key, test_x.shape,
        nsteps=1_000, traj=False,
    )
    # Taos: todo: parametrize kstart in config
    kstart = 3
    err = pmetric(stochastic_samples[:, kstart:], T_long[kstart:], ds.integrate)[0]

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

        if isinstance(cfg.model, cs.ModelDiffusion):
            train_diffusion(cfg)
        elif isinstance(cfg.model, cs.ModelFlowMatching):
            train_flow_matching(cfg)
        else:
            raise ValueError(f'Unknown model: {cfg.model}')


def get_run_dir(hydra_init=HYDRA_INIT, commit=True):
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
