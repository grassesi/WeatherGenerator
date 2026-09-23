# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Data side of prediction masking (optional_target_sst_masking.md, C and D).

Covers name resolution of `mask_predictions`, the `valid` mask that get_target_coords builds in
prediction row order, its storage on StreamData, and the valid-time restamp of the base_idx
fallback window (M7). No real dataset is opened: readers and samplers are faked or built bare.
"""

import numpy as np
import pytest
import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets.data_reader_base import ReaderData, TimeWindowHandler
from weathergen.datasets.multi_stream_data_sampler import (
    MultiStreamDataSampler,
    resolve_mask_cols,
    restamp_to_window,
)
from weathergen.datasets.stream_data import StreamData
from weathergen.datasets.tokenizer_masking import TokenizerMasking

HL = 1
T0 = np.datetime64("2020-01-01T00:00", "ns")

# --- resolve_mask_cols -------------------------------------------------------------------------


def test_resolve_mask_cols_maps_names_to_target_columns():
    info = {"name": "Ocean", "mask_predictions": ["sst", "ci"]}
    assert resolve_mask_cols(info, ["t2m", "ci", "sst"]) == [2, 1]


@pytest.mark.parametrize("info", [{"name": "Ocean"}, {"name": "Ocean", "mask_predictions": []}])
def test_resolve_mask_cols_is_empty_when_nothing_is_masked(info):
    assert resolve_mask_cols(info, ["sst"]) == []


def test_resolve_mask_cols_raises_on_unknown_channel():
    info = {"name": "Ocean", "mask_predictions": ["sst", "sstt"]}
    with pytest.raises(ValueError, match=r"Ocean.*sstt.*Available: \['t2m', 'sst'\]"):
        resolve_mask_cols(info, ["t2m", "sst"])


# --- get_target_coords -------------------------------------------------------------------------


def _target_rdata(n: int = 200) -> IOReaderData:
    """n points on the sphere; column 0 is NaN on points with lat > 30 (recognisable by coords),
    column 1 is NaN on every 7th point, column 2 is finite."""
    rng = np.random.default_rng(0)
    lats = rng.uniform(-89.0, 89.0, n).astype(np.float32)
    lons = rng.uniform(-179.0, 179.0, n).astype(np.float32)
    data = rng.normal(size=(n, 3)).astype(np.float32)
    data[lats > 30.0, 0] = np.nan
    data[::7, 1] = np.nan
    return IOReaderData(
        coords=np.stack([lats, lons], axis=1),
        geoinfos=np.zeros((n, 0), dtype=np.float32),
        data=data,
        datetimes=np.full(n, T0),
    )


def _target_coords(mask_cols, keep_cells: bool = True):
    tok = TokenizerMasking(HL, masker=None)
    stream_info = {"stream_id": 0, "token_size": 8}
    rdata = _target_rdata()
    (token_data,) = tok.get_tokens_windows(stream_info, [rdata], False)
    cell_mask = np.full(12 * 4**HL, keep_cells, dtype=bool)
    time_win = (T0, T0 + np.timedelta64(6, "h"))
    return tok.get_target_coords(
        stream_info, rdata, token_data, time_win, cell_mask, mask_cols=mask_cols
    )


def test_valid_is_false_exactly_at_masked_nan_rows_in_prediction_order():
    coords_local, _, coords_raw, datetimes, valid = _target_coords([0])

    n = coords_raw.shape[0]
    assert n == 200
    assert valid.dtype == torch.bool
    assert valid.shape == (n, 3)
    assert coords_local.shape[0] == n and len(datetimes) == n
    # rows line up with coords_raw: column 0 is invalid exactly on the lat > 30 points
    lats = torch.as_tensor(coords_raw)[:, 0]
    assert torch.equal(valid[:, 0], ~(lats > 30.0))
    assert (~valid[:, 0]).any() and valid[:, 0].any()
    # column 1 holds NaNs but is not masked, column 2 has none
    assert valid[:, 1].all() and valid[:, 2].all()


@pytest.mark.parametrize("mask_cols", [None, []])
def test_valid_is_empty_without_mask_cols(mask_cols):
    *_, valid = _target_coords(mask_cols)
    assert valid.dtype == torch.bool
    assert valid.shape == (0, 0)


def test_valid_on_the_empty_tokenization_path():
    # no cell kept: tokenize_apply_mask_target takes its empty-return path
    *_, valid = _target_coords([0], keep_cells=False)
    assert valid.dtype == torch.bool
    assert valid.shape == (0, 3)


# --- StreamData --------------------------------------------------------------------------------


def test_stream_data_stores_and_moves_target_valid():
    num_cells = 12 * 4**HL
    sd = StreamData(0, 1, 2, num_cells)
    assert all(v.shape == (0, 0) and v.dtype == torch.bool for v in sd.target_valid)

    valid = torch.tensor([[True, False], [True, True]])
    sd.add_target_coords(
        "val", 1, torch.zeros((2, 4)), torch.zeros(num_cells), False, target_valid=valid
    )
    assert torch.equal(sd.target_valid[1], valid)
    assert sd.target_valid[0].shape == (0, 0)

    sd.to_device("cpu")
    assert torch.equal(sd.target_valid[1], valid)
    assert sd.target_valid[0].shape == (0, 0)


# --- M7: restamp of the base_idx fallback ------------------------------------------------------


def test_restamp_to_window_shifts_by_the_window_offset_and_keeps_intra_window_offsets():
    times = T0 + np.array([0, 1, 5], dtype="timedelta64[h]")
    rdata = IOReaderData(
        coords=np.zeros((3, 2), dtype=np.float32),
        geoinfos=np.zeros((3, 0), dtype=np.float32),
        data=np.zeros((3, 1), dtype=np.float32),
        datetimes=times.copy(),
    )
    out = restamp_to_window(rdata, T0, T0 + np.timedelta64(48, "h"))

    np.testing.assert_array_equal(out.datetimes, times + np.timedelta64(48, "h"))
    # the input is not mutated: readers may hand out cached arrays
    np.testing.assert_array_equal(rdata.datetimes, times)


class _FakeReader:
    """Serves a 4-point target window only for idx <= last_idx, hourly from the window start."""

    stream_info = {"name": "Ocean"}

    def __init__(self, tw: TimeWindowHandler, last_idx: int):
        self.tw = tw
        self.last_idx = last_idx

    def _window(self, idx):
        if idx > self.last_idx:
            return ReaderData.empty(num_data_fields=1, num_geo_fields=0)
        n = 4
        start = np.datetime64(self.tw.window(idx).start, "ns")
        return ReaderData(
            coords=np.zeros((n, 2), dtype=np.float32),
            geoinfos=np.zeros((n, 0), dtype=np.float32),
            data=np.arange(n, dtype=np.float32)[:, None],
            datetimes=start + np.arange(n) * np.timedelta64(1, "h"),
        )

    get_source = get_target = _window

    def normalize_source_channels(self, x):
        return x

    normalize_target_channels = normalize_geoinfos = normalize_source_channels


def test_fallback_window_carries_the_forecast_steps_valid_times():
    step = np.timedelta64(24, "h")
    tw = TimeWindowHandler(T0, T0 + np.timedelta64(30, "D"), step, step)

    s = object.__new__(MultiStreamDataSampler)
    s.time_window_handler = tw
    s.rng = np.random.default_rng(0)
    s.output_offset = 1
    s.time_step = step
    s.step_timedelta = step
    s.healpix_level = HL

    base_idx = 3
    # the target exists up to base_idx only, so forecast steps 1 and 2 fall back to it
    reader = _FakeReader(tw, last_idx=base_idx)
    _, output_data = s._get_data_windows(base_idx, 2, 1, [reader])

    base = reader.get_target(base_idx).datetimes
    for fstep, rdata in zip((1, 2), output_data, strict=True):
        shift = tw.window(base_idx + fstep).start - tw.window(base_idx).start
        assert rdata.is_spoof
        np.testing.assert_array_equal(rdata.datetimes, base + shift)
