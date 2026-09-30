# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest

from weathergen.datasets.stream_data import spoof
from weathergen.datasets.utils import coords_to_hpyidxs

START = np.datetime64("2009-01-01T00:00", "ms")


def _encoder_chunks(model_level: int) -> int:
    """Chunks `EncoderModule.assimilate_local_project_chunked` splits the cells into."""
    return 2 if model_level <= 5 else 8


# ForcingInput spoofs at level 5 whatever the model's level; the samplers spoof at the model's
@pytest.mark.parametrize("model_level", range(1, 8))
@pytest.mark.parametrize("spoof_level", ["5", "model"])
def test_a_spoof_fills_every_encoder_chunk(model_level, spoof_level):
    level = 5 if spoof_level == "5" else model_level
    rdata = spoof(level, START, geoinfo_size=0, num_channels=1)

    cells = coords_to_hpyidxs(model_level, rdata.coords[:, 0], rdata.coords[:, 1])
    num_cells = 12 * 4**model_level
    chunk = num_cells // _encoder_chunks(model_level)
    filled = np.unique(np.asarray(cells) // chunk)

    # an empty chunk is skipped, which under FSDP leaves this rank a collective short
    np.testing.assert_array_equal(filled, np.arange(_encoder_chunks(model_level)))


def test_a_spoof_is_the_same_on_every_call():
    a = spoof(5, START, geoinfo_size=0, num_channels=1)
    b = spoof(5, START, geoinfo_size=0, num_channels=1)
    np.testing.assert_array_equal(a.coords, b.coords)
