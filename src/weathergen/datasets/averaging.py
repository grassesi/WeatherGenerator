import typing

import numpy as np
import pandas as pd

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TIndex,
    WrappedDataReader,
)
from weathergen.datasets.geoinfo import recompute_geoinfos


class AveragingReader(DataReaderTimestep, WrappedDataReader):
    def __init__(self, wrapped_reader: DataReaderTimestep):
        self._wrapped_reader = wrapped_reader

        super().__init__(
            self._wrapped_reader.time_window_handler,
            self._wrapped_reader.stream_info,
            self._wrapped_reader.data_start_time,
            self._wrapped_reader.data_end_time,
            self._wrapped_reader.period,
        )

        self.source_channels = self._wrapped_reader.source_channels
        self.target_channels = self._wrapped_reader.target_channels
        self.geoinfo_channels = self._wrapped_reader.geoinfo_channels
        self.source_idx = self._wrapped_reader.source_idx
        self.target_idx = self._wrapped_reader.target_idx
        self.geoinfo_idx = self._wrapped_reader.geoinfo_idx
        self.target_channel_weights = self._wrapped_reader.target_channel_weights

        self.mean = self._wrapped_reader.mean
        self.stdev = self._wrapped_reader.stdev
        self.mean_geoinfo = self._wrapped_reader.mean_geoinfo
        self.stdev_geoinfo = self._wrapped_reader.stdev_geoinfo

    @typing.override
    def length(self) -> int:
        return self._wrapped_reader.length()

    @typing.override
    def init_empty(self) -> None:
        self._wrapped_reader.init_empty()

    @typing.override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Calculate Averages if multiple datapoints per gridpoint exist.

        Averages are calculated on all source/target channels; the geoinfos follow
        `recompute_geoinfos`, which owns that policy.

        Semantics: a window [start, end) averages every datapoint the wrapped reader
        returns for it -- which is exactly the grid times in [start, end) -- and serves the
        result as one datapoint stamped at `start`. This is the same contract an un-averaged
        stream on the same grid honours.
        """
        rdata = self._wrapped_reader._get(idx, channels_idx)
        if rdata.is_empty():  # Reading empty is a normal state, not a defect.
            return rdata

        stamp = rdata.datetimes.min()
        rows = rdata.datetimes == stamp

        data = (
            pd.DataFrame(
                np.concat([rdata.coords, rdata.data], axis=1),
                columns=["lat", "lon", *channels_idx],
                # groupby() implicitly sorts, unaligning data and coordinates.
            )
            .groupby(["lat", "lon"], sort=False)
            .mean()
        )

        return ReaderData(
            coords=rdata.coords[rows],
            geoinfos=recompute_geoinfos(
                rdata.geoinfos, rdata.coords, rdata.datetimes, self.geoinfo_channels, stamp
            ),
            data=data[channels_idx].values,
            datetimes=rdata.datetimes[rows],
            is_spoof=rdata.is_spoof,
        )
