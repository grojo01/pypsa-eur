# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Solves linear optimal dispatch using the capacities of a previous capacity
expansion in rule [solve_network][], on the same snapshots and resolution.

Loads `networks/solved_{horizon}.nc`, fixes the optimal capacities, and
re-solves dispatch only (optionally with rolling horizon). Note that
`load_shedding` is incompatible with operational solving, as its components
already exist in the solved network.

With rolling horizon, stores and storage units start from their level before
the first snapshot in the solved network, a failed slice raises an error, and
the objective is the sum over the slices (overlapping snapshots count in both
slices). Annual global constraints apply to every slice in full.
"""

import logging
from functools import partial

import numpy as np
import pandas as pd
import pypsa

from scripts._benchmark import memory_logger
from scripts._helpers import configure_logging, set_scenario_config
from scripts.solve_network import (
    collect_kwargs,
    create_optimization_model,
    extra_functionality,
    prepare_network,
)

logger = logging.getLogger(__name__)

STORAGE_LEVELS = [
    ("Store", "e", "e_initial", "e_cyclic"),
    (
        "StorageUnit",
        "state_of_charge",
        "state_of_charge_initial",
        "cyclic_state_of_charge",
    ),
]


def initial_storage_levels(n: pypsa.Network) -> dict[str, pd.Series]:
    """
    Storage levels before the first snapshot of a solved network.

    Cyclic stores and storage units start at their level of the last snapshot,
    reduced by the standing loss of the first snapshot as in the cyclic energy
    balance. Non-cyclic ones keep their initial level.
    """
    first = n.snapshots[0]
    hours = n.snapshot_weightings.stores.loc[first]
    levels = {}
    for c, level, initial, cyclic in STORAGE_LEVELS:
        static = n.c[c].static
        loss = n.get_switchable_as_dense(c, "standing_loss").loc[first]
        last = n.c[c].dynamic[level].iloc[-1].reindex(static.index)
        levels[c] = static[initial].where(~static[cyclic], last * (1 - loss) ** hours)
    return levels


def set_initial_storage_levels(n: pypsa.Network, levels: dict[str, pd.Series]) -> None:
    for c, _, initial, _ in STORAGE_LEVELS:
        n.c[c].static[initial] = levels[c]


class SliceCheck:
    """
    Wrap `extra_functionality` to check every rolling-horizon slice.

    `optimize_with_rolling_horizon` only logs a warning for a failed slice and
    keeps no objective. `extra_functionality` is called for every slice; each
    call reads the solved model of the previous slice, so a failed slice raises
    before the next one starts. Call `check()` once more after the last slice.
    """

    def __init__(self, extra_functionality):
        self.extra_functionality = extra_functionality
        self.model = None
        self.objectives = []

    def __call__(self, n: pypsa.Network, snapshots: pd.Index) -> None:
        self.check()
        self.model, self.snapshots = n.model, snapshots
        self.extra_functionality(n, snapshots)

    def check(self) -> None:
        m, self.model = self.model, None
        if m is None:
            return
        if m.status != "ok":
            raise RuntimeError(
                f"Rolling-horizon slice [{self.snapshots[0]}:{self.snapshots[-1]}] "
                f"failed with status {m.status} and condition "
                f"{m.termination_condition}."
            )
        self.objectives.append(m.objective.value)


def warn_annual_limits(n: pypsa.Network) -> None:
    """Warn about global constraints that sum over the snapshots."""
    types = ["primary_energy", "operational_limit", "co2_atmosphere"]
    glcs = n.global_constraints.index[n.global_constraints.type.isin(types)]
    if not glcs.empty:
        logger.warning(
            "Rolling horizon: every slice gets the full annual limit of the "
            f"global constraints {list(glcs)}, not a share of it."
        )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_operations_network",
            configfiles="config/test/config.electricity.yaml",
            horizon=2030,
        )

    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)

    cf_solving = snakemake.params.solving["options"]
    cf_operations = snakemake.params.solving["operations"]
    np.random.seed(cf_solving.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)
    planning_horizons = snakemake.wildcards["horizon"]

    rolling_horizon = cf_operations["rolling_horizon"]

    n.optimize.fix_optimal_capacities()
    if rolling_horizon:
        levels = initial_storage_levels(n)

    prepare_network(
        n,
        config=snakemake.config,
        solve_opts=cf_solving,
        foresight=snakemake.params.foresight,
        planning_horizons=planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
        limit_max_growth=snakemake.params["sector"]["limit_max_growth"],
        rolling_horizon=rolling_horizon,
    )
    if rolling_horizon:
        set_initial_storage_levels(n, levels)
        warn_annual_limits(n)

    logging_frequency = snakemake.config.get("solving", {}).get(
        "mem_logging_frequency", 30
    )

    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=logging_frequency
    ) as mem:
        if rolling_horizon:
            logger.info("Solving operations network with rolling horizon...")
            all_kwargs, _ = collect_kwargs(
                snakemake.config,
                snakemake.params.solving,
                planning_horizons,
                log_fn=snakemake.log.solver,
                mode="rolling_horizon",
                horizon=cf_operations["horizon"],
                overlap=cf_operations["overlap"],
            )
            n.config = snakemake.config
            n.params = snakemake.params
            slices = SliceCheck(
                partial(
                    extra_functionality,
                    planning_horizons=planning_horizons,
                    snakemake=snakemake,
                )
            )
            all_kwargs["extra_functionality"] = slices
            n.optimize.optimize_with_rolling_horizon(**all_kwargs)
            slices.check()
            n._objective = sum(slices.objectives)
        else:
            logger.info("Solving operations network...")
            model_kwargs, solve_kwargs = collect_kwargs(
                snakemake.config,
                snakemake.params.solving,
                planning_horizons,
                log_fn=snakemake.log.solver,
                mode="single",
            )
            create_optimization_model(
                n,
                config=snakemake.config,
                params=snakemake.params,
                model_kwargs=model_kwargs,
                solve_kwargs=solve_kwargs,
                planning_horizons=planning_horizons,
                snakemake=snakemake,
            )
            n.optimize.solve_model(**solve_kwargs)

    logger.info(f"Maximum memory usage: {mem.mem_usage}")

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output.network)
