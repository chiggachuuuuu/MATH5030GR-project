from heston_mc.params import HestonParams


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
