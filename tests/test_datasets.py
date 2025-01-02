import pytest
from omegaconf import OmegaConf

from fixtures import init_hydra_cfg, engine
from userfm import cs

from userfm import datasets
from userdiffusion import ode_datasets as datasets_old


@pytest.mark.parametrize('overrides', [
    ['+experiment=Lorenz', 'dataset.trajectory_count=20', 'dataset.time_step_count=100', 'model=ModelDiffusion'],
    ['+experiment=FitzHughNagumo', 'dataset.trajectory_count=20', 'dataset.time_step_count=667', 'model=ModelDiffusion'],
])
def test_datasets_deterministic_with_rng_seed(engine, overrides):
    cfg = init_hydra_cfg('config', overrides)
    with cs.orm.Session(engine) as session:
        cfg = cs.instantiate_and_insert_config(session, OmegaConf.to_container(cfg))
        dss = []
        for _ in range(2):
            dss.append(datasets.get_dataset(cfg.dataset, rng_seed=cfg.rng_seed))
        ds1, ds2 = dss
        assert len(ds1) == len(ds2)
        for i in range(len(ds1)):
            assert (ds1[i][0][0] == ds2[i][0][0]).all()
            assert (ds1[i][0][1] == ds2[i][0][1]).all()
            assert (ds1[i][1] == ds2[i][1]).all()


@pytest.mark.parametrize('overrides', [
    ['+experiment=Lorenz', 'dataset.trajectory_count=20', 'dataset.time_step_count=100', 'model=ModelDiffusion'],
    ['+experiment=FitzHughNagumo', 'dataset.trajectory_count=20', 'dataset.time_step_count=667', 'model=ModelDiffusion'],
])
def test_datasets_equal(engine, overrides):
    cfg = init_hydra_cfg('config', overrides)
    with cs.orm.Session(engine) as session:
        cfg = cs.instantiate_and_insert_config(session, OmegaConf.to_container(cfg))
        ds = datasets.get_dataset(cfg.dataset, rng_seed=cfg.rng_seed)
        ds_old = datasets_old.get_dataset(cfg.dataset, rng_seed=cfg.rng_seed)
        assert len(ds) == len(ds_old)
        for i in range(len(ds)):
            assert (ds[i][0][0] == ds_old[i][0][0]).all()
            assert (ds[i][0][1] == ds_old[i][0][1]).all()
            assert (ds[i][1] == ds_old[i][1]).all()
