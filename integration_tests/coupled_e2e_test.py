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
reader, whether a request against real asymmetric cadences resolves to the windows it should,
or whether the exchange reports itself honestly. Each of those has been wrong in a way a green
unit suite did not notice, which is why the provenance counts below assert numbers derived by
hand from the configuration rather than anything read off a run.

Must run on a GPU machine.

    uv run pytest ./integration_tests/coupled_e2e_test.py
"""

import logging
import re
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


# Where each component's forcing windows must come from, derived by hand from `coupled.yml`:
# one sample, two 24 h chunks, an atmosphere at 6 h against an ocean at 24 h, both at the
# default lag of one of the consumer's own windows.
#
# Ocean consumes ATMO. Its chunk is one 24 h step, so it makes two requests, each on a 24 h
# window lagged 24 h: [t0, t0+24h) and [t0+24h, t0+48h). Four 6 h atmospheric windows start
# inside each, so eight source windows in all. The one at t0 is the initialization window and
# is primed from the atmosphere's own data; the other seven are predictions.
#
# Atmo consumes OCEAN. Four 6 h steps per chunk, eight requests, each on a 6 h window lagged
# 6 h. No 24 h oceanic window starts inside a 6 h request except the first of each chunk, so
# six of the eight are held from the window covering them. The first chunk's four resolve to
# the init window and are primed; the second chunk's four to the ocean's first prediction.
PROVENANCE = {
    "Ocean": ("'ATMO' of 'Ocean'", 2, 7, 1, 0, 0),
    "Atmo": ("'OCEAN' of 'Atmo'", 8, 4, 4, 0, 6),
}


@pytest.mark.parametrize("component", sorted(PROVENANCE))
def test_every_forcing_window_is_accounted_for(coupled_run, component):
    """The load-bearing assertion about the numbers rather than the wiring.

    Every way this exchange can fail ends at the same climatological spoof behind the same
    debug line, with every wiring assertion still green: a request resolved on the producer's
    index space instead of the consumer's, a window the producer has not emitted, a lag that
    reaches past what is stored. The counts distinguish them, and they are derivable by hand
    from the configuration rather than read off a run.
    """
    who, requests, predicted, primed, disk, held = PROVENANCE[component]
    expected = (
        f"Forcing {who}: {requests} request(s) -> {predicted} predicted, {primed} primed, "
        f"{disk} from disk, 0 unresolved ({held} held from a covering window)."
    )

    assert expected in coupled_run, (
        "provenance line missing or wrong; the run printed: "
        + "; ".join(line for line in coupled_run.splitlines() if "Forcing " in line)
    )


def test_no_forcing_window_falls_through_to_a_spoof(coupled_run):
    """An unresolved window is a silent substitution of climatology for a partner's state."""
    assert "unresolved" not in coupled_run.replace("0 unresolved", "")


def test_the_slow_component_gathers_the_fast_one(coupled_run):
    """C2: one request on a 24 h window must resolve to the four emissions inside it.

    The ocean's own `average_window` then collapses them, exactly as it did in training. An
    exchange that returns one window per request instead leaves it averaging a single row --
    a quarter of the forcing it was trained on, and no error anywhere. Read off the run rather
    than off the table above, so this says something the count assertions do not.
    """
    line = next(
        line for line in coupled_run.splitlines() if "Forcing 'ATMO' of 'Ocean'" in line
    )
    requests = int(re.search(r"(\d+) request", line).group(1))
    resolved = sum(
        int(n) for n in re.findall(r"(\d+) (?:predicted|primed|from disk)", line)
    )

    assert resolved == 4 * requests, f"four source windows per request, got: {line}"


def test_each_direction_declares_its_lag(coupled_run):
    """D4: the lag is validated against the step order, and both bounds are reported."""
    assert "forcing_lag 24:00:00, at least 18:00:00 required" in coupled_run
    assert "forcing_lag 06:00:00, at least 06:00:00 required" in coupled_run
    assert "Step order, from couplings-file declaration order: ['Atmo', 'Ocean']." in coupled_run


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
