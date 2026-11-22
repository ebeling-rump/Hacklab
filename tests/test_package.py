"""Every module imports and every file the experiments read is in place."""

import importlib
import pkgutil

import numpy as np
import pytest

import a1slip
from a1slip.paths import DATA, FIGURES, RESULTS

MODULES = [m.name for m in pkgutil.walk_packages(a1slip.__path__, "a1slip.")]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", ["slip.npz", "slip_balanced.npz",
                                  "collision_env_A.npz", "collision_env_B.npz"])
def test_datasets_present(name):
    assert (DATA / name).is_file()


def test_slip_table_shape():
    d = np.load(DATA / "slip.npz", allow_pickle=True)
    n = len(d["slip_next"])
    assert d["features"].shape == (n, 51) and d["history"].shape == (n, 11)   # 62 numbers per row
    assert set(np.unique(d["slip_next"])) <= {0, 1}


def test_readme_figures_present():
    for name in ("ice_governor.png", "ice_patrol.png", "outdoor_maps.png", "fall_roc.png"):
        assert (FIGURES / name).is_file()
    assert (RESULTS / "ice_governor.json").is_file()
