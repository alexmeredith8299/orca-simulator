"""M2: torque-free motion conserves inertial angular momentum; the quaternion stays unit."""

import numpy as np

from expand3.params import Params, ParamView
from expand3.vehicle.attitude import AttitudeState, inertia, step, total_momentum_inertial
from expand3.vehicle.wheels import WheelArray


def test_torque_free_conservation(params: Params) -> None:
    for sat in ("SAT-1", "SAT-5"):
        view = ParamView(params, sat)
        inertia_b, w = inertia(view), WheelArray(view)
        s = AttitudeState(q=np.array([1.0, 0, 0, 0]), omega=np.array([0.01, -0.02, 0.005]),
                          h=np.zeros(4))
        dt = 1.0
        for _ in range(60):  # spin up the wheels against the body
            s, _, _ = step(inertia_b, w, s, np.array([0.03, 0.02, -0.04]), dt)
        h0 = total_momentum_inertial(inertia_b, w, s)
        assert np.linalg.norm(s.h) > 1.0
        for _ in range(3600):
            s, u, _ = step(inertia_b, w, s, np.zeros(3), dt)
            assert np.all(u == 0)
        h1 = total_momentum_inertial(inertia_b, w, s)
        assert np.linalg.norm(h1 - h0) / np.linalg.norm(h0) < 1e-9
        assert abs(np.linalg.norm(s.q) - 1) < 1e-12
