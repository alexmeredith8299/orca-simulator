"""Every fault target in faults.yaml must name a parameter ParamView can resolve."""

import pytest

from expand3.params import Params, ParamView


@pytest.mark.parametrize("fid", [f"F{i}" for i in range(1, 11)])
def test_fault_path_resolves(params: Params, fid: str) -> None:
    fault = params.faults.faults[fid]
    view = ParamView(params, fault.vehicle)
    for path in fault.paths:
        assert view.has(path), path
