"""
pytest configuration for the gpu_corrcal accuracy suite.

Run from src/gpu_corrcal/tests:

    pytest -s                         # all accuracy tests, default sizes
    pytest -s -m "not slow"           # skip the large grids
    pytest -s -k logdet               # one kernel
    pytest -s --sizes 18x32,32x48     # choose antenna grids
    pytest -s --neig 3 --nsrc 5       # pin the template parameters
    pytest -s --refresh-ref           # recompute cached CPU references

-s shows the per-block error table for every case; without it, the table
appears only for failures.
"""

import sys
from pathlib import Path

import pytest

# Make accuracy_helpers / cpu_reference importable from the per-kernel folders.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Antenna grids (rows, cols). n_row = n_ant * (n_ant - 1) after the Re/Im split.
FAST_SIZES = [(6, 8), (18, 32)]          # 2,256 and 331,200 rows
SLOW_SIZES = [(32, 48)]                  # 2,357,760 rows

DEFAULT_NEIG = [1, 2, 3, 4, 8, 12, 16, 20]
DEFAULT_NSRC = [1, 3, 8, 20]


def pytest_addoption(parser):
    g = parser.getgroup("gpu_corrcal accuracy")
    g.addoption("--sizes", default=None,
                help="comma list of RxC grids, e.g. 6x8,18x32 (overrides defaults)")
    g.addoption("--neig", default=None, help="comma list of n_eig values")
    g.addoption("--nsrc", default=None, help="comma list of n_src values")
    g.addoption("--seed", type=int, default=12, help="data seed")
    g.addoption("--refresh-ref", action="store_true",
                help="ignore and overwrite cached CPU references")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: large grids, CPU reference is expensive")


def _ints(s):
    return [int(x) for x in s.split(",")]


def pytest_generate_tests(metafunc):
    opt = metafunc.config.getoption

    if "rc" in metafunc.fixturenames:
        if opt("sizes"):
            params = [tuple(int(v) for v in s.split("x")) for s in opt("sizes").split(",")]
        else:
            params = FAST_SIZES + [pytest.param(s, marks=pytest.mark.slow) for s in SLOW_SIZES]
        metafunc.parametrize("rc", params, ids=lambda rc: f"{rc[0]}x{rc[1]}")

    if "n_eig" in metafunc.fixturenames:
        vals = _ints(opt("neig")) if opt("neig") else DEFAULT_NEIG
        metafunc.parametrize("n_eig", vals, ids=lambda v: f"eig{v}")

    if "n_src" in metafunc.fixturenames:
        vals = _ints(opt("nsrc")) if opt("nsrc") else DEFAULT_NSRC
        metafunc.parametrize("n_src", vals, ids=lambda v: f"src{v}")


@pytest.fixture(scope="session")
def seed(request):
    return request.config.getoption("seed")


@pytest.fixture(scope="session")
def refresh(request):
    return request.config.getoption("refresh_ref")