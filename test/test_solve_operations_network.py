# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import logging

import pandas as pd
import pypsa
import pytest

from scripts.solve_operations_network import (
    SliceCheck,
    initial_storage_levels,
    set_initial_storage_levels,
    warn_annual_limits,
)


@pytest.fixture(scope="function")
def solved() -> pypsa.Network:
    """
    One bus, generator 2 MW, load 3-1-1-3 MW and a cyclic store.

    The store covers the deficit in the first and last snapshot, so it enters
    the period with at least 1 MWh.
    """
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2013-01-01", periods=4, freq="h"))
    n.add("Bus", "electricity")
    n.add("Load", "load", bus="electricity", p_set=[3.0, 1.0, 1.0, 3.0])
    n.add("Generator", "gen", bus="electricity", p_nom=2.0, marginal_cost=1.0)
    n.add("Store", "cyclic", bus="electricity", e_nom=10.0, e_cyclic=True)
    n.add(
        "Store",
        "lossy",
        bus="electricity",
        e_nom=10.0,
        e_cyclic=True,
        standing_loss=0.1,
    )
    n.add("Store", "fixed", bus="electricity", e_nom=10.0, e_initial=2.0)
    n.add("StorageUnit", "unit", bus="electricity", p_nom=1.0, max_hours=10.0)
    n.optimize(solver_name="highs")
    return n


def rolling_horizon(n: pypsa.Network) -> SliceCheck:
    n.stores.e_cyclic = False
    n.storage_units.cyclic_state_of_charge = False
    slices = SliceCheck(lambda n, snapshots: None)
    n.optimize.optimize_with_rolling_horizon(
        horizon=2, solver_name="highs", extra_functionality=slices
    )
    slices.check()
    return slices


def test_initial_storage_levels(solved: pypsa.Network) -> None:
    levels = initial_storage_levels(solved)
    e = solved.stores_t.e.iloc[-1]
    assert levels["Store"]["cyclic"] == pytest.approx(e["cyclic"])
    assert levels["Store"]["lossy"] == pytest.approx(e["lossy"] * 0.9)
    assert levels["Store"]["fixed"] == 2.0
    soc = solved.storage_units_t.state_of_charge.iloc[-1]
    assert levels["StorageUnit"]["unit"] == pytest.approx(soc["unit"])


def test_slices_sum_objective(solved: pypsa.Network) -> None:
    set_initial_storage_levels(solved, initial_storage_levels(solved))
    slices = rolling_horizon(solved)
    assert len(slices.objectives) == 2
    cost = solved.generators_t.p.sum().sum() * 1.0
    assert sum(slices.objectives) == pytest.approx(cost)


def test_failed_slice_raises(solved: pypsa.Network) -> None:
    solved.stores.e_initial = 0.0
    solved.storage_units.state_of_charge_initial = 0.0
    with pytest.raises(RuntimeError, match="failed with status"):
        rolling_horizon(solved)


def test_warn_annual_limits(solved: pypsa.Network, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        warn_annual_limits(solved)
    assert not caplog.records
    solved.add(
        "GlobalConstraint",
        "CO2Limit",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        constant=0.0,
    )
    with caplog.at_level(logging.WARNING):
        warn_annual_limits(solved)
    assert "CO2Limit" in caplog.text
