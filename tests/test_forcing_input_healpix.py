"""ForcingInput builds its forcing StreamData on the component's own HEALPix level.

It used to be hardcoded to level 5, so a level-3 ocean model built forcing samples with
12288 cell slots against a tokenizer binning into 768.
"""

import types

import numpy as np
import pytest

from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.model.forcing import ForcingInput

H24 = np.timedelta64(24, "h")


def _handler() -> TimeWindowHandler:
    return TimeWindowHandler(
        np.datetime64("2023-01-01T00:00"), np.datetime64("2023-02-01T00:00"), H24, H24
    )


@pytest.mark.parametrize("level", [3, 4, 5])
def test_the_forcing_grid_follows_the_component_level(level):
    tokenizer = types.SimpleNamespace(healpix_level=level)
    forcings = ForcingInput("validation", _handler(), {}, tokenizer, healpix_level=level)

    assert forcings.healpix_lvl == level
    assert forcings.num_healpix_cells == 12 * 4**level


def test_a_tokenizer_on_another_level_is_refused():
    tokenizer = types.SimpleNamespace(healpix_level=5)

    with pytest.raises(ValueError, match="healpix_level 3, but its tokenizer bins at 5"):
        ForcingInput("validation", _handler(), {}, tokenizer, healpix_level=3)
