# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Contract of rebase_innermost: rebuild a reader stack on a different innermost reader."""

import numpy as np

from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    ReaderData,
    WrappedDataReader,
    rebase_innermost,
)


class Leaf(DataReaderBase):
    """A bare datasource: the thing at the bottom of a stack."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.source_channels = ["a"]
        self.source_idx = [0]
        self.target_channels = []
        self.target_idx = []
        self.geoinfo_channels = []
        self.geoinfo_idx = []
        self.target_channel_weights = []

    def length(self) -> int:
        return 1

    def _get(self, idx, channels_idx) -> ReaderData:
        raise NotImplementedError


class Wrapper(DataReaderBase, WrappedDataReader):
    """A decorator carrying derived state, as the real wrappers do."""

    def __init__(self, wrapped, name: str) -> None:
        self._wrapped_reader = wrapped
        self.name = name
        for attr in (
            "source_channels", "target_channels", "geoinfo_channels",
            "source_idx", "target_idx", "geoinfo_idx", "target_channel_weights",
        ):
            setattr(self, attr, getattr(wrapped, attr))
        # stands in for the parsed schedules and channel tables the real wrappers derive at
        # construction and cannot rebuild from the instance
        self.derived = f"derived-from-{wrapped.tag if isinstance(wrapped, Leaf) else wrapped.name}"

    def length(self) -> int:
        return self._wrapped_reader.length()

    def _get(self, idx, channels_idx) -> ReaderData:
        raise NotImplementedError


def _stack() -> Wrapper:
    return Wrapper(Wrapper(Leaf("disk"), "inner"), "outer")


def innermost(reader):
    while isinstance(reader, WrappedDataReader):
        reader = reader._wrapped_reader
    return reader


def test_innermost_reader_is_replaced():
    stack = _stack()

    rebased = rebase_innermost(stack, lambda base: Leaf(f"coupled({base.tag})"))

    assert innermost(rebased).tag == "coupled(disk)"


def test_the_original_stack_is_untouched():
    """The readers are shared with the sampler, so a rebase that mutates corrupts the batch."""
    stack = _stack()

    rebase_innermost(stack, lambda base: Leaf("coupled"))

    assert innermost(stack).tag == "disk"


def test_every_wrapper_is_a_distinct_object():
    """A shared wrapper would let the rebased stack's state leak back into the original."""
    stack = _stack()

    rebased = rebase_innermost(stack, lambda base: Leaf("coupled"))

    assert rebased is not stack
    assert rebased._wrapped_reader is not stack._wrapped_reader


def test_derived_state_survives_the_copy():
    """copy.copy is used precisely because this state cannot be rebuilt from the instance."""
    stack = _stack()

    rebased = rebase_innermost(stack, lambda base: Leaf("coupled"))

    assert rebased.derived == stack.derived
    assert rebased.name == "outer"


def test_a_bare_reader_is_replaced_outright():
    leaf = Leaf("disk")

    rebased = rebase_innermost(leaf, lambda base: Leaf(f"coupled({base.tag})"))

    assert rebased.tag == "coupled(disk)"


def test_the_real_wrappers_are_rebasable():
    from weathergen.datasets.averaging import AveragingReader
    from weathergen.datasets.elevation import ElevatingReader
    from weathergen.datasets.upsampling import UpsamplingReader

    for cls in (AveragingReader, ElevatingReader, UpsamplingReader):
        assert issubclass(cls, WrappedDataReader), f"{cls.__name__} cannot be rebased"
        assert np.all(True)
