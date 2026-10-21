# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Hydro pumping configuration, added to the schema as a soft-fork config update.

See docs in https://pypsa-eur.readthedocs.io/en/latest/validation_dev/#soft_fork_ext
"""

from pydantic import Field, model_validator

from scripts.lib.validation.config._base import ConfigModel, ConfigUpdater
from scripts.lib.validation.config._schema import ConfigSchema


class HydroPumpingConfig(ConfigModel):
    """Configuration for `hydro_pumping` settings."""

    enable: bool = Field(
        False,
        description="Add extendable pumping capacity to existing reservoir hydro (carrier `hydro`).",
    )
    countries: list[str] = Field(
        default_factory=list,
        description="Two-letter country codes whose `hydro` clusters may receive pumping capacity.",
    )
    total_capacity: float | None = Field(
        None,
        description="Total pumping capacity in MW over all `hydro` clusters in `countries`. If set, it is fixed and `capital_cost` must be 0; if null, the optimiser chooses it.",
        ge=0,
    )
    capital_cost: float = Field(
        0.0,
        description="Annualised capital cost of pumping capacity in EUR/MW/a. Only allowed if `total_capacity` is null. With 0, pumping is free and the optimiser builds as much as is useful.",
        ge=0,
    )

    @model_validator(mode="after")
    def check_countries(self):
        if self.enable and not self.countries:
            raise ValueError(
                "hydro_pumping: `countries` must not be empty if `enable` is true."
            )
        return self

    @model_validator(mode="after")
    def check_capital_cost(self):
        if self.total_capacity is not None and self.capital_cost > 0:
            raise ValueError(
                "hydro_pumping: `capital_cost` must be 0 if `total_capacity` is "
                "set; a fixed capacity is not priced."
            )
        return self


class HydroPumpingConfigUpdater(ConfigUpdater):
    # empty name writes into config/config.default.yaml, the file Snakemake reads
    name: str = ""
    docs_url: str | None = None

    def update(self) -> type[ConfigSchema]:
        return self._apply_updates(
            hydro_pumping=(
                HydroPumpingConfig,
                Field(
                    default_factory=HydroPumpingConfig,
                    description="Extendable pumping capacity at existing reservoir hydro.",
                ),
            )
        )
