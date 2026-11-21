# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Add extendable pumping capacity to existing reservoir hydro.

In the countries listed in `hydro_pumping: countries`, each `hydro`
StorageUnit is replaced by a store-link topology: an upper reservoir
fed by the inflow, a turbine link, an extendable pump link and a lower
reservoir. The lower reservoir is filled by the turbines and the spillage
of the upper one and supplies the pumps; a free release lets water leave
it. The StorageUnit's attributes are taken over unchanged, so without
pumping capacity the optimisation problem is the same. Everywhere else the
network is left untouched.
"""

import logging

import pandas as pd
import pypsa

from scripts.add_electricity import add_missing_carriers

logger = logging.getLogger(__name__)


def add_hydro_pumping(n: pypsa.Network, config: dict, costs: pd.DataFrame) -> None:
    """
    Replace `hydro` StorageUnits in `config["countries"]` by store and links.

    Per StorageUnit with power P (MW), turbine efficiency eta_T
    (`efficiency_dispatch`), `max_hours` h and the share s =
    `config["lower_reservoir_share"]`:

    - Store ``{name} reservoir`` with ``e_nom`` = h * P (MWh), cyclic.
    - Generator ``{name} inflow`` with ``p_nom`` = 1 MW and ``p_min_pu`` =
      ``p_max_pu`` = inflow in MW, so the inflow is fed in fully. With
      ``p_nom`` = 1 MW the solver option ``clip_p_max_pu`` clips the inflow
      exactly as it clips StorageUnit inflow.
    - Store ``{name} lower reservoir`` with ``e_nom`` = s * h * P (MWh),
      cyclic.
    - Link ``{name} spill`` reservoir -> lower reservoir with ``p_nom`` =
      1 MW and ``p_max_pu`` = inflow, so spillage is bounded by the inflow
      as the curtailment of the StorageUnit inflow is.
    - Link ``{name} turbine`` reservoir -> AC with ``p_nom`` = P / eta_T
      (bus0 side), ``efficiency`` = eta_T, ``marginal_cost`` and
      ``capital_cost`` (EUR/MW/a) scaled by eta_T to bus0, ``p_max_pu`` as
      the StorageUnit's. ``bus2`` = lower reservoir with ``efficiency2`` =
      1: the turbined water reaches the lower reservoir.
    - Link ``{name} pump`` AC -> reservoir, extendable up to P (MW), with
      ``efficiency`` = eta_RT / eta_T, where eta_RT is the PHS round-trip
      efficiency from `costs`, and ``capital_cost`` = `config["capital_cost"]`
      (EUR/MW/a), which the config validation keeps at 0 if
      `config["total_capacity"]` is set. ``bus2`` = lower reservoir with
      ``efficiency2`` = -eta_RT / eta_T: per MWh arriving in the reservoir,
      one MWh leaves the lower reservoir. ``bus0`` stays AC, so ``p_nom``
      remains in MW of electricity.
    - Generator ``{name} release`` on the lower reservoir with ``p_min_pu``
      = -1, ``p_max_pu`` = 0 and ``p_nom`` = P / eta_T + max inflow (MW),
      at no cost: water leaves the lower reservoir to the river, so a full
      lower reservoir never blocks the turbine or the spillage.

    All reservoir quantities are in the unit of the StorageUnit's state of
    charge, the electricity value of the water.

    If `config["total_capacity"]` (MW) is set, a GlobalConstraint row
    ``hydro pump total capacity`` is added as placeholder for the constraint
    from `add_hydro_pumping_constraint`.

    Only static StorageUnit attributes are taken over; `attach_hydro` sets
    no time-varying ones besides ``inflow``.
    """
    if not config["enable"]:
        return

    su = n.storage_units.query("carrier == 'hydro'")
    su = su[su.bus.map(n.buses.country).isin(config["countries"])]
    if su.empty:
        raise ValueError(
            f"hydro_pumping: no `hydro` storage units in {config['countries']}."
        )

    names = su.index
    eff_turbine = su.efficiency_dispatch
    eff_pump = costs.at["PHS", "efficiency"] / eff_turbine
    logger.info(
        f"Adding hydro pumping to {len(su)} reservoirs in {config['countries']}."
    )

    add_missing_carriers(
        n,
        [
            "hydro reservoir",
            "hydro inflow",
            "hydro turbine",
            "hydro pump",
            "hydro lower reservoir",
            "hydro spill",
            "hydro release",
        ],
    )
    reservoir = names + " reservoir"
    lower = names + " lower reservoir"
    inflow = n.storage_units_t.inflow[names]
    n.add(
        "Bus",
        reservoir,
        carrier="hydro reservoir",
        location=su.bus.values,
        x=n.buses.x[su.bus].values,
        y=n.buses.y[su.bus].values,
        country=n.buses.country[su.bus].values,
    )
    n.add(
        "Store",
        reservoir,
        bus=reservoir,
        carrier="hydro reservoir",
        e_nom=(su.max_hours * su.p_nom).values,
        e_cyclic=True,
    )
    n.add(
        "Generator",
        names + " inflow",
        bus=reservoir,
        carrier="hydro inflow",
        p_nom=1.0,
        p_min_pu=inflow.set_axis(names + " inflow", axis=1),
        p_max_pu=inflow.set_axis(names + " inflow", axis=1),
    )
    n.add(
        "Bus",
        lower,
        carrier="hydro lower reservoir",
        location=su.bus.values,
        x=n.buses.x[su.bus].values,
        y=n.buses.y[su.bus].values,
        country=n.buses.country[su.bus].values,
    )
    n.add(
        "Store",
        lower,
        bus=lower,
        carrier="hydro lower reservoir",
        e_nom=(config["lower_reservoir_share"] * su.max_hours * su.p_nom).values,
        e_cyclic=True,
    )
    n.add(
        "Link",
        names + " spill",
        bus0=reservoir,
        bus1=lower,
        carrier="hydro spill",
        p_nom=1.0,
        p_max_pu=inflow.set_axis(names + " spill", axis=1),
    )
    n.add(
        "Generator",
        names + " release",
        bus=lower,
        carrier="hydro release",
        p_nom=(su.p_nom / eff_turbine + inflow.max()).values,
        p_min_pu=-1.0,
        p_max_pu=0.0,
    )
    n.add(
        "Link",
        names + " turbine",
        bus0=reservoir,
        bus1=su.bus.values,
        bus2=lower,
        carrier="hydro turbine",
        p_nom=(su.p_nom / eff_turbine).values,
        p_max_pu=su.p_max_pu.values,
        efficiency=eff_turbine.values,
        efficiency2=1.0,
        marginal_cost=(su.marginal_cost * eff_turbine).values,
        capital_cost=(su.capital_cost * eff_turbine).values,
    )
    n.add(
        "Link",
        names + " pump",
        bus0=su.bus.values,
        bus1=reservoir,
        bus2=lower,
        carrier="hydro pump",
        p_nom_extendable=True,
        p_nom_max=su.p_nom.values,
        efficiency=eff_pump.values,
        efficiency2=-eff_pump.values,
        capital_cost=config["capital_cost"],
    )
    n.remove("StorageUnit", names)

    if config["total_capacity"] is not None:
        if config["total_capacity"] > su.p_nom.sum():
            raise ValueError(
                f"hydro_pumping: `total_capacity` of {config['total_capacity']} MW "
                f"exceeds the {su.p_nom.sum():.0f} MW of `hydro` turbines in "
                f"{config['countries']}."
            )
        n.add(
            "GlobalConstraint",
            "hydro pump total capacity",
            type="hydro_pumping",
            carrier_attribute="hydro pump",
            sense="==",
            constant=config["total_capacity"],
        )


def add_hydro_pumping_constraint(n: pypsa.Network) -> None:
    """
    Fix the summed ``p_nom`` of all ``hydro pump`` links to `total_capacity`.

    Called from `extra_functionality` if `add_hydro_pumping` added the
    GlobalConstraint row. PyPSA's ``tech_capacity_expansion_limit`` is not
    used because PyPSA 1.2.4 skips it without investment periods. The model
    constraint carries the row's name, so PyPSA stores its shadow price in
    ``n.global_constraints.mu``: the change in total system cost per
    additional MW of pumping capacity (EUR/MW/a for one full year). It is
    negative while more pumping lowers the cost.

    Pumps with fixed ``p_nom`` count with their capacity. If no pump is
    extendable, e.g. when the network is solved again after
    ``fix_optimal_capacities``, the fixed capacities already hold the total
    and no constraint is added.
    """
    name = "hydro pump total capacity"
    if name not in n.global_constraints.index:
        return
    pumps = n.links.query("carrier == 'hydro pump'")
    ext = pumps.index[pumps.p_nom_extendable]
    if ext.empty:
        return
    fixed = pumps.p_nom[~pumps.p_nom_extendable].sum()
    lhs = n.model["Link-p_nom"].sel(name=ext).sum()
    n.model.add_constraints(
        lhs == n.global_constraints.at[name, "constant"] - fixed,
        name=f"GlobalConstraint-{name}",
    )
