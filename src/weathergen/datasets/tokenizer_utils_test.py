# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the target token layout constants.

`TIMES_WIDTH` is a constant rather than a call into the encoders, so it is pinned here
instead: the encoders are what define it, and this is the one place that says so.
"""

import numpy as np

from weathergen.datasets.tokenizer_utils import (
    TIMES_WIDTH,
    encode_times_source,
    encode_times_target,
)

TIMES = np.array(["2023-01-01T00", "2023-01-01T03"], dtype="datetime64[ns]")
WINDOW = (np.datetime64("2023-01-01T00"), np.datetime64("2023-01-01T06"))


def test_target_time_encoding_is_times_width_wide():
    assert encode_times_target(TIMES, WINDOW).shape == (len(TIMES), TIMES_WIDTH)


def test_source_time_encoding_is_times_width_wide():
    assert encode_times_source(TIMES, WINDOW).shape == (len(TIMES), TIMES_WIDTH)
