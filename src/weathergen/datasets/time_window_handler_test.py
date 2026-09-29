# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for TimeWindowHandler.align_first_prediction."""

import numpy as np
import pytest

from weathergen.datasets.data_reader_base import TimeWindowHandler, shifted

START = np.datetime64("2023-01-01T00:00")
END = np.datetime64("2023-12-31T00:00")
H6 = np.timedelta64(6, "h")
H24 = np.timedelta64(24, "h")


def test_unaligned_handler_starts_at_start_date():
    handler = TimeWindowHandler(START, END, H24, H24)

    assert handler.t_start == START
    assert handler.window(0).start == START
    assert handler.first_prediction_start == START + H24


@pytest.mark.parametrize("step", [H6, H24])
def test_aligned_handler_predicts_its_first_window_at_start_date(step):
    """The initialization window moves back one step, so the first prediction starts at T."""

    handler = TimeWindowHandler(START, END, step, step, align_first_prediction=True)

    assert handler.t_start == START - step
    assert handler.window(0).end == START, "the initialization window ends where T begins"
    assert handler.window(1).start == START
    assert handler.first_prediction_start == START


def test_components_on_different_steps_share_their_first_prediction():
    atmo = TimeWindowHandler(START, END, H6, H6, align_first_prediction=True)
    ocean = TimeWindowHandler(START, END, H24, H24, align_first_prediction=True)

    assert atmo.t_start == START - H6
    assert ocean.t_start == START - H24
    assert atmo.first_prediction_start == ocean.first_prediction_start == START


def test_the_shift_counts_forecast_offset_windows_and_leaves_the_end():
    handler = TimeWindowHandler(START, END, H6, H6, align_first_prediction=True, forecast_offset=2)

    assert handler.t_start == START - 2 * H6
    assert handler.t_end == END
    assert handler.first_prediction_start == START


def test_offset_zero_does_not_shift():
    handler = TimeWindowHandler(START, END, H6, H6, align_first_prediction=True, forecast_offset=0)

    assert handler.t_start == START


def test_a_lagged_timeline_does_not_shift_twice():
    """`shifted` rebuilds from the resolved start, so only the lag moves it."""

    aligned = TimeWindowHandler(START, END, H24, H24, align_first_prediction=True)
    lagged = shifted(aligned, np.timedelta64(0, "h"))

    assert lagged.t_start == aligned.t_start
    assert lagged.window(1).start == aligned.window(1).start


def test_the_flag_is_logged():
    assert "first prediction at" in str(
        TimeWindowHandler(START, END, H6, H6, align_first_prediction=True)
    )
    assert "first prediction at" not in str(TimeWindowHandler(START, END, H6, H6))
