import numpy as np

from heston_mc.params import default_heston_params, default_mc_config
from heston_mc.simulation import HestonModelSimulator


def test_simulation_output_shape_and_validity():
    params = default_heston_params()
    config = default_mc_config()
    
    config.n_paths = 100
    config.n_steps = 50
    
    simulator = HestonModelSimulator(params, config)
    S, v = simulator.simulate()
    
    assert S.shape == (100, 51)
    assert v.shape == (100, 51)
    
    assert not np.any(np.isnan(S))
    assert not np.any(np.isnan(v))
    assert np.all(S >= 0.0)