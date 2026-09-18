# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the value semantics of ReaderData."""

import dataclasses

import numpy as np
import pytest

from weathergen.datasets.data_reader_base import ReaderData

NUM_POINTS = 6
NUM_CHANNELS = 2
NUM_GEOINFOS = 3


def make_rdata(num_points: int = NUM_POINTS, is_spoof: bool = False) -> ReaderData:
    return ReaderData(
        coords=np.arange(num_points * 2, dtype=np.float32).reshape(num_points, 2),
        geoinfos=np.arange(num_points * NUM_GEOINFOS, dtype=np.float32).reshape(
            num_points, NUM_GEOINFOS
        ),
        data=np.arange(num_points * NUM_CHANNELS, dtype=np.float32).reshape(
            num_points, NUM_CHANNELS
        ),
        datetimes=np.array(
            [np.datetime64("2023-01-01T00:00") + np.timedelta64(i, "h") for i in range(num_points)]
        ),
        is_spoof=is_spoof,
    )


def test_fields_cannot_be_rebound():
    rdata = make_rdata()

    with pytest.raises(dataclasses.FrozenInstanceError):
        rdata.data = np.zeros((1, NUM_CHANNELS), dtype=np.float32)


def test_shuffle_does_not_modify_the_original():
    rdata = make_rdata()
    original = rdata.data.copy()

    shuffled = rdata.shuffle(np.random.default_rng(0), shuffle=True, num_subset=-1)

    assert shuffled is not rdata
    np.testing.assert_array_equal(rdata.data, original)


def test_shuffle_keeps_all_rows_when_only_shuffling():
    rdata = make_rdata()

    shuffled = rdata.shuffle(np.random.default_rng(0), shuffle=True, num_subset=-1)

    assert shuffled.len() == rdata.len()
    # every row survives, as a set
    assert sorted(shuffled.data[:, 0].tolist()) == sorted(rdata.data[:, 0].tolist())


def test_shuffle_keeps_the_fields_of_a_row_together():
    rdata = make_rdata()

    shuffled = rdata.shuffle(np.random.default_rng(1), shuffle=True, num_subset=4)

    assert shuffled.len() == 4
    for row in range(shuffled.len()):
        # row r of the original has data[r, 0] == 2 * r and coords[r, 0] == 2 * r
        assert shuffled.coords[row, 0] == shuffled.data[row, 0]


def test_shuffle_subsets_without_shuffling_preserves_order():
    rdata = make_rdata()

    subset = rdata.shuffle(np.random.default_rng(2), shuffle=False, num_subset=3)

    assert subset.len() == 3
    assert (np.diff(subset.datetimes) > np.timedelta64(0)).all()


def test_shuffle_is_a_noop_when_nothing_is_asked_for():
    rdata = make_rdata()

    assert rdata.shuffle(None, shuffle=False, num_subset=-1) is rdata


def test_shuffle_of_empty_data_is_a_noop():
    rdata = ReaderData.empty(NUM_CHANNELS, NUM_GEOINFOS)

    assert rdata.shuffle(np.random.default_rng(0), shuffle=True, num_subset=-1) is rdata


def test_remove_nan_does_not_modify_the_original():
    rdata = make_rdata()
    rdata.coords[2, 0] = np.nan
    original_len = rdata.len()

    cleaned = rdata.remove_nan_coords_and_geoinfos()

    assert cleaned.len() == original_len - 1
    assert rdata.len() == original_len


def test_remove_nan_drops_rows_with_nan_geoinfos():
    rdata = make_rdata()
    rdata.geoinfos[1, 2] = np.nan

    cleaned = rdata.remove_nan_coords_and_geoinfos()

    assert cleaned.len() == rdata.len() - 1
    assert not np.isnan(cleaned.geoinfos).any()


def test_transforms_carry_is_spoof_through():
    rdata = make_rdata(is_spoof=True)

    assert rdata.shuffle(np.random.default_rng(0), shuffle=True, num_subset=3).is_spoof
    assert rdata.remove_nan_coords_and_geoinfos().is_spoof


def test_copy_detaches_every_array():
    rdata = make_rdata()

    copied = rdata.copy()
    copied.data[0, 0] = -1.0
    copied.coords[0, 0] = -1.0
    copied.geoinfos[0, 0] = -1.0

    assert rdata.data[0, 0] == 0.0
    assert rdata.coords[0, 0] == 0.0
    assert rdata.geoinfos[0, 0] == 0.0


def test_copy_preserves_values_and_flags():
    rdata = make_rdata(is_spoof=True)

    copied = rdata.copy()

    np.testing.assert_array_equal(copied.data, rdata.data)
    np.testing.assert_array_equal(copied.coords, rdata.coords)
    np.testing.assert_array_equal(copied.datetimes, rdata.datetimes)
    assert copied.is_spoof == rdata.is_spoof


def test_replace_is_the_supported_way_to_change_a_field():
    rdata = make_rdata()

    normalized = dataclasses.replace(rdata, data=rdata.data * 2.0)

    np.testing.assert_array_equal(normalized.data, rdata.data * 2.0)
    np.testing.assert_array_equal(normalized.coords, rdata.coords)
