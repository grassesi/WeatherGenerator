# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""End-to-end test of a two-sided coupled rollout.

Two components, each predicting one stream and forced by the other's, stepped in one
interleaved loop. Trains both halves from scratch, then couples them.

What this covers that the unit tests cannot: the unit tests drive the Coupler with fakes, so
they say nothing about whether a real reader stack survives being rebased onto a coupling
reader, whether the levelling wrapper picks the right conversion for real cadences, or whether
the exchange reports itself honestly. Each of those has been wrong in a way a green unit suite
did not notice.

Must run on a GPU machine.

    uv run pytest ./integration_tests/coupled_e2e_test.py
"""

import logging
import shutil
import zipfile
from pathlib import Path

import pytest

from weathergen.run_train import main

logger = logging.getLogger(__name__)

WEATHERGEN_HOME = Path(__file__).parent.parent
MINI_EPOCH = 0

try:
    from git import Repo

    COMMIT = Repo(search_parent_directories=False).head.object.hexsha[:5]
except Exception:
    COMMIT = "unknown"

ATMO_ID = f"cpl_e2e_atmo_{COMMIT}"
OCEAN_ID = f"cpl_e2e_ocean_{COMMIT}"
COUPLED_ID = f"cpl_e2e_run_{COMMIT}"


def _clean(*run_ids: str) -> None:
    for rid in run_ids:
        # logs too: a run appends, so a log left by an earlier attempt would satisfy assertions
        # this run never earned
        for sub in ("results", "models", "logs"):
            shutil.rmtree(WEATHERGEN_HOME / sub / rid, ignore_errors=True)


@pytest.fixture(scope="module")
def components() -> tuple[str, str]:
    """Both halves, trained from scratch. Module-scoped: training dominates the runtime."""
    _clean(ATMO_ID, OCEAN_ID, COUPLED_ID)
    for side, run_id in (("atmo", ATMO_ID), ("ocean", OCEAN_ID)):
        main(
            [
                "train",
                f"--base-config={WEATHERGEN_HOME}/integration_tests/small1.yml",
                f"--config={WEATHERGEN_HOME}/integration_tests/coupled_{side}.yml",
                "--run-id",
                run_id,
            ]
        )
        chkpts = sorted((WEATHERGEN_HOME / "models" / run_id).glob("*.chkpt"))
        assert chkpts, f"{side} trained no checkpoint; a coupled run has nothing to load"
    return ATMO_ID, OCEAN_ID


@pytest.fixture(scope="module")
def coupled_run(components) -> str:
    """The coupled rollout, returning what it wrote to its own log.

    Read from `logs/<run_id>/log.txt` rather than captured in-process: the run reconfigures
    logging during setup, which drops any handler installed beforehand, and `caplog` is
    function-scoped so it cannot serve a module-scoped fixture anyway. The log file is also what
    a real run is judged by, so the test reads the same evidence an operator would.
    """
    atmo_id, ocean_id = components
    main(
        [
            "coupled_inference",
            f"{WEATHERGEN_HOME}/integration_tests/coupled.yml",
            f"Atmo={atmo_id}@{MINI_EPOCH}",
            f"Ocean={ocean_id}@{MINI_EPOCH}",
            "--run-id",
            COUPLED_ID,
        ]
    )
    log = WEATHERGEN_HOME / "logs" / COUPLED_ID / "log.txt"
    assert log.is_file(), f"no log at {log}: the run left nothing to check"
    return log.read_text()


def test_both_directions_exchange(coupled_run):
    """The load-bearing assertion: a two-sided coupling must exchange in both directions.

    A consumer that does not read the partner's stream as a dynamic forcing is a silent no-op,
    so "declared" and "exchanging" are different numbers and only the second one matters.
    """
    substituted = [
        line for line in coupled_run.splitlines() if "is forced by component" in line
    ]

    assert len(substituted) == 2, f"expected both directions live, got: {substituted}"
    assert any("'Atmo' is forced by component 'Ocean'" in line for line in substituted)
    assert any("'Ocean' is forced by component 'Atmo'" in line for line in substituted)


def test_the_summary_agrees_with_the_substitutions(coupled_run):
    """The summary is derived from the substitution record, so it cannot drift from it.

    It has twice reported a live exchange as dead -- once from a hardcoded milestone string,
    once from testing the outermost reader's type through a wrapper.
    """
    assert "2 coupling(s) declared, 2 with a consumer, 2 actually exchanging." in coupled_run
    assert "has no effect" not in coupled_run, "no coupling should be reported dead"


def test_each_direction_levels_its_own_cadence(coupled_run):
    """The two directions need different levelling, which is why this pair is asymmetric.

    Atmo -> Ocean: the atmosphere produces 6-hourly and the ocean reads that stream 6-hourly,
    so nothing is levelled -- the ocean's own `average_window` collapses the four samples in its
    24 h window, exactly as it did in training.

    Ocean -> Atmo: the ocean produces 24-hourly but the atmosphere was trained on a 6-hourly
    field, so the coupling upsamples. Getting this backwards, or levelling when nothing needs
    it, corrupts the forcing without failing anything else.
    """
    assert "Stream 'ATMO': producer and consumer both sample every 06:00:00" in coupled_run
    assert "no levelling needed" in coupled_run
    assert "Stream 'OCEAN': producer samples every 24:00:00" in coupled_run
    assert "upsampling." in coupled_run
    assert "averaging down" not in coupled_run


def test_each_component_writes_its_own_streams(coupled_run):
    """One store, disjoint by construction: output.streams is each component's produced streams."""
    stores = sorted((WEATHERGEN_HOME / "results" / COUPLED_ID).glob("validation_*.zip"))

    assert stores, "a coupled run that writes nothing has not been verified by anything"
    names = zipfile.ZipFile(stores[0]).namelist()
    streams = {n.split("/")[1] for n in names if n.count("/") >= 2 and not n.startswith("zarr")}

    assert {"ATMO", "OCEAN"} <= streams, f"both components must write; got {streams}"


def test_the_rollout_runs_to_completion(coupled_run):
    """The wrapper's exit code lies, so completion is judged by the run's own last word."""
    assert "Finished coupled inference run with id" in coupled_run
