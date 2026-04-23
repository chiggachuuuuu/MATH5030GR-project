
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.heston_mc.experiments import (
    run_baseline_experiment,
    run_path_sensitivity_experiment,
    run_timestep_sensitivity_experiment,
    run_parameter_robustness_experiment
)

if __name__ == "__main__":
    print("Starting Numerical Experiments...")
    run_baseline_experiment()
    run_path_sensitivity_experiment()
    run_timestep_sensitivity_experiment()
    run_parameter_robustness_experiment()
    print("All experiments completed successfully.")