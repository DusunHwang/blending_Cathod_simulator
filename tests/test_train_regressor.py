import os
import subprocess
import sys
import importlib


def test_train_regressor_outputs(tmp_path):
    subprocess.check_call([sys.executable, 'train_regressor.py', '--epochs', '2'])
    assert os.path.exists('models/final_model.pt')
    assert os.path.exists('logs/train_log.csv')
    assert os.path.exists('reports/eval.md')
    assert os.path.exists('plots/train_vs_test.png')


def test_simulate_blend_runs():
    if importlib.util.find_spec('pybamm') is None:
        import pytest
        pytest.skip('pybamm not installed')
    subprocess.check_call([sys.executable, 'simulate_blend.py', '--blend', '0.5'])
