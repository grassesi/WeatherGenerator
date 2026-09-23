# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""`mask_predictions`: the step that NaNs predictions at invalid target points."""

import pytest
import torch

from weathergen.model.model import mask_predictions


def _pred(n: int, n_channels: int, n_ens: int = 2) -> torch.Tensor:
    return torch.arange(n_ens * n * n_channels, dtype=torch.float32).reshape(n_ens, n, n_channels)


def test_masks_exactly_the_invalid_entries_across_samples():
    lens = [3, 2]
    pred = _pred(sum(lens), 2)
    valid = [
        torch.tensor([[True, True], [False, True], [True, True]]),
        torch.tensor([[True, False], [True, True]]),
    ]

    out = mask_predictions(pred, valid, lens)

    expected_nan = torch.zeros(5, 2, dtype=torch.bool)
    expected_nan[1, 0] = True  # sample 0, row 1, channel 0
    expected_nan[3, 1] = True  # sample 1, row 0, channel 1
    for ens in range(pred.shape[0]):
        assert torch.equal(torch.isnan(out[ens]), expected_nan)
        assert torch.equal(out[ens][~expected_nan], pred[ens][~expected_nan])


def test_empty_validity_is_a_no_op():
    lens = [3, 2]
    pred = _pred(sum(lens), 1)
    empty = torch.zeros((0, 0), dtype=torch.bool)

    assert mask_predictions(pred, [empty, empty], lens) is pred
    assert mask_predictions(pred, [None, None], lens) is pred


def test_sample_without_validity_is_left_whole():
    lens = [2, 2]
    pred = _pred(sum(lens), 1, n_ens=1)
    valid = [torch.tensor([[False], [True]]), torch.zeros((0, 0), dtype=torch.bool)]

    out = mask_predictions(pred, valid, lens)

    assert torch.isnan(out[0, 0, 0])
    assert not torch.isnan(out[0, 1:]).any()


def test_misaligned_validity_is_refused():
    with pytest.raises(AssertionError, match="Prediction validity has shape"):
        mask_predictions(_pred(3, 1), [torch.ones((2, 1), dtype=torch.bool)], [3])
