"""The simulation reproduces saved results exactly (no TabPFN needed for these runs)."""

import json

import mujoco
import numpy as np
import pytest

from a1slip.paths import RESULTS

pytestmark = pytest.mark.sim


@pytest.mark.parametrize("seed,speed", [(0, 1.0), (1, 0.3)])
def test_ice_course_reproduces_saved_run(seed, speed):
    from a1slip.ice.patches import B, PatchSensors, build_patch_model, steer, walk

    saved = {(r["seed"], r["strategy"]): r for r in json.loads((RESULTS / "ice_governor.json").read_text())}
    ref = saved[(seed, f"fixed {speed}")]
    model = build_patch_model(seed)
    leg = walk(model, PatchSensors(model), [B], lambda f, h, d, g: steer(d, g, speed), max_time=60)["legs"][0]
    assert leg["time"] == pytest.approx(ref["time"], abs=0.01)
    assert leg["slip_s"] == pytest.approx(ref["slip_s"], abs=0.01)


def test_episode_produces_62_numbers_per_row():
    from a1slip.sim.slip_dataset import A1Sensors, build_model, labels, run_episode

    model = build_model()
    ep = run_episode(model, mujoco.MjData(model), A1Sensors(model), np.random.default_rng(0),
                     mu=0.15, slope=12.0, azimuth=0.3)
    assert ep["features"].shape[1] + ep["history"].shape[1] == 62
    assert len(labels(ep)[0]) == len(ep["features"])
