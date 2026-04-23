import os

import pytest

from heston_mc.experiments import (
    run_baseline_experiment,
    run_path_sensitivity_experiment,
    setup_results_directory
)


def test_setup_results_directory(tmp_path):
    test_dir = tmp_path / "test_results"
    path = setup_results_directory(str(test_dir))
    
    assert os.path.exists(path)
    assert os.path.isdir(path)


def test_baseline_experiment_execution(capsys):
    try:
        run_baseline_experiment()
    except Exception as e:
        pytest.fail(f"run_baseline_experiment raised {type(e).__name__} unexpectedly!")
    
    captured = capsys.readouterr()
    assert "Variance Swap" in captured.out
    assert "Variance Call Option" in captured.out


def test_path_sensitivity_file_generation():
    output_dir = "results"
    csv_file = os.path.join(output_dir, "path_sensitivity.csv")
    plot_file = os.path.join(output_dir, "se_convergence_paths.png")
    
    run_path_sensitivity_experiment()
    
    assert os.path.exists(csv_file)
    assert os.path.exists(plot_file)