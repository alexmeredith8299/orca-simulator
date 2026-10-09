"""Every fault target in faults.yaml must name a parameter ParamView can resolve."""

import pytest

from expand3.params import Params, ParamView

# faults.yaml (verbatim from the spec) names state arrays that bus_a.yaml spells differently
# (wheels.k / wheels.c vs k_nominal / c_nominal_per_s) or does not define yet
# (eps.string_health). The M7 fault injector decides the mapping; drop the xfail then.
UNRESOLVED = {"F2", "F3", "F4"}


@pytest.mark.parametrize("fid", [f"F{i}" for i in range(1, 11)])
def test_fault_path_resolves(params: Params, fid: str, request: pytest.FixtureRequest) -> None:
    if fid in UNRESOLVED:
        request.applymarker(pytest.mark.xfail(strict=True, reason="path mapping deferred to M7"))
    fault = params.faults.faults[fid]
    view = ParamView(params, fault.vehicle)
    for path in fault.paths:
        assert view.has(path), path
