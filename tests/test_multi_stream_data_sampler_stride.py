# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the init-stride math in MultiStreamDataSampler (init_stride_design.md).

check_samples() and _check_stride_config() are exercised directly on a bare instance
(object.__new__, attributes set by hand) rather than through __init__, which needs a real
dataset. What is under test is arithmetic and an assertion, neither of which touches I/O.
"""

import numpy as np
import pytest

from weathergen.datasets.data_reader_base import TimeIndexRange
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler


def make_sampler(
    *,
    index_end: int,
    sample_stride: int,
    batch_size: int = 1,
    samples_per_mini_epoch: int,
    world_size: int = 1,
    repeat_data: bool = False,
) -> MultiStreamDataSampler:
    """A bare instance carrying only what check_samples() reads."""
    s = object.__new__(MultiStreamDataSampler)
    s.index_range = TimeIndexRange(start=0, end=index_end)
    s.time_step = np.timedelta64(0, "h")
    s.output_offset = 0
    s.len_timedelta = np.timedelta64(0, "h")
    s.step_timedelta = np.timedelta64(1, "h")
    s.sample_stride = sample_stride
    s.batch_size = batch_size
    s.samples_per_mini_epoch = samples_per_mini_epoch
    s.world_size = world_size
    s.repeat_data = repeat_data
    return s


def test_available_samples_counts_the_inclusive_on_grid_range():
    """Indices [0, max_index] at stride k number floor(max_index/k) + 1, not floor(...)."""
    # max_index == index_range.end here (zero forecast horizon), 10 // 3 + 1 == 4 on-grid
    # indices (0, 3, 6, 9); samples_per_mini_epoch (100) forces the clamp to available - 1.
    s = make_sampler(index_end=10, sample_stride=3, samples_per_mini_epoch=100)
    s.check_samples(fsm=0)

    assert s.len == 3


def test_stride_one_matches_the_previous_behaviour():
    """At stride 1 every index [0, max_index] is visited: max_index + 1 of them."""
    s = make_sampler(index_end=10, sample_stride=1, samples_per_mini_epoch=100)
    s.check_samples(fsm=0)

    assert s.len == 10


@pytest.mark.parametrize("stride,expected_len", [(3, 3), (1, 10)])
def test_n_initializations_at_stride_k_land_exactly_k_windows_apart(stride, expected_len):
    """A sampler unit test: N initializations at stride k land exactly k windows apart."""
    s = make_sampler(index_end=10, sample_stride=stride, samples_per_mini_epoch=100)
    s.check_samples(fsm=0)

    assert s.len == expected_len


def test_shuffle_and_stride_together_are_rejected():
    """S3: a stride walks a shuffled permutation and enforces no spacing at all."""
    with pytest.raises(AssertionError, match="init_stride_fsteps"):
        MultiStreamDataSampler._check_stride_config(shuffle=True, sample_stride=2)


def test_shuffle_alone_or_stride_alone_is_fine():
    MultiStreamDataSampler._check_stride_config(shuffle=True, sample_stride=1)
    MultiStreamDataSampler._check_stride_config(shuffle=False, sample_stride=5)


def test_stride_below_one_is_rejected():
    with pytest.raises(AssertionError, match="init_stride_fsteps"):
        MultiStreamDataSampler._check_stride_config(shuffle=False, sample_stride=0)
