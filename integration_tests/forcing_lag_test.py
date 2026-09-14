# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A training run at a non-default forcing lag.

The lag is a property of the *model*, not of a run: the dynamic-forcing response is trained at
one lag, so the same lag has to hold through finetuning and inference
(`forcing_lag_design.md` L3). What makes that true rather than merely intended is that the key
lives on the stream in the component's own config and is therefore written into the checkpoint,
which is what a later inference reads -- editing a YAML afterwards changes nothing.

This is the only test that runs a non-default lag at all: every deployed stream takes the
default, so without it the key is exercised in unit tests and nowhere else.

Must run on a GPU machine.

    uv run pytest ./integration_tests/forcing_lag_test.py
"""

import json
import logging
import shutil
from pathlib import Path

import pytest

from weathergen.run_train import main

logger = logging.getLogger(__name__)

WEATHERGEN_HOME = Path(__file__).parent.parent

# Not a multiple of the atmosphere's 6 h window step, so the resolved value cannot coincide
# with the default by accident and the backwards quantisation is what the rollout runs on.
LAG = "09:00:00"

try:
    from git import Repo

    COMMIT = Repo(search_parent_directories=False).head.object.hexsha[:5]
except Exception:
    COMMIT = "unknown"

RUN_ID = f"lag_e2e_{COMMIT}"


@pytest.fixture(scope="module")
def trained() -> str:
    for sub in ("results", "models", "logs"):
        shutil.rmtree(WEATHERGEN_HOME / sub / RUN_ID, ignore_errors=True)

    main(
        [
            "train",
            f"--base-config={WEATHERGEN_HOME}/integration_tests/small1.yml",
            f"--config={WEATHERGEN_HOME}/integration_tests/coupled_atmo.yml",
            "--run-id",
            RUN_ID,
            "--options",
            f"streams.OCEAN.forcing_lag={LAG}",
        ]
    )
    return RUN_ID


def test_the_resolved_lag_is_logged_at_setup(trained):
    """An operator has to be able to read the lag a run actually used off its log."""
    log = (WEATHERGEN_HOME / "logs" / trained / "log.txt").read_text()

    assert f"Forcing stream 'OCEAN' is sampled {LAG} before the window it forces." in log


def test_the_lag_reaches_the_checkpoint(trained):
    """L3: inference loads the config stored with the checkpoint, not the YAML in config/.

    So a lag that trains but is not saved is a lag the rollout will silently not use.
    """
    saved = sorted((WEATHERGEN_HOME / "models" / trained).glob(f"model_{trained}_*.json"))
    assert saved, "the run stored no resolved config to check"

    cfg = json.loads(saved[-1].read_text())

    assert cfg["streams"]["OCEAN"]["forcing_lag"] == LAG
