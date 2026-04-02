from heston_mc.params import HestonParams, MonteCarloConfig


def test_heston_params_validate():
    params = HestonParams(
        s0=100.0,
        v0=0.04,
        r=0.03,
        q=0.0,
        kappa=2.0,
        theta=0.04,
        sigma=0.5,
        rho=-0.7,
    )
    params.validate()


def test_mc_config_validate():
    cfg = MonteCarloConfig(
        maturity=1.0,
        n_steps=252,
        n_paths=10000,
        seed=42,
        periods_per_year=252,
    )
    cfg.validate()
