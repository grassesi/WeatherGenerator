import typing

import numpy as np
import pandas as pd

from weathergen.datasets.data_reader_base import DataReaderTimestep, ReaderData, TIndex

AVERAGING_GEOINFOS = ["z", "insolation"]


class AveragingReader(DataReaderTimestep):
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

        # only average selected geoinfo channels
        self.averaging_geoinfo_idx = [
            idx
            for idx, channel in enumerate(self.geoinfo_channels)
            if channel in AVERAGING_GEOINFOS
        ]
        self.averaging_geoinfos = [
            channel for channel in self.geoinfo_channels if channel in AVERAGING_GEOINFOS
        ]  # construct unique column labels
        self.non_averaging_geoinfo_idx = [
            idx for idx, _ in enumerate(self.geoinfo_idx) if idx not in self.averaging_geoinfo_idx
        ]

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

        Averages are calculated on all source/target channels and on selected
        geoinfo channels (eg. insolation, z). Static features and time features
        are taken from last datapoint in the interval. The new data thus has the
        following semantics: data in interval (<start>, <end>) => data averaged over the last <interval len> hours at time <end>.
        """
        rdata = self._wrapped_reader._get(idx, channels_idx)
        max_time_idx = np.argwhere(rdata.datetimes == rdata.datetimes.max())

        data = (
            pd.DataFrame(
                np.concat(
                    [rdata.coords, rdata.geoinfos[:, self.averaging_geoinfo_idx], rdata.data],
                    axis=1,
                ),
                columns=["lat", "lon", *self.averaging_geoinfos, *channels_idx],
                # groupby() implicitly sorts, unaligning data and coordinates: the coords,
                # geoinfos and datetimes below are taken unsorted via max_time_idx.
            )
            .groupby(["lat", "lon"], sort=False)
            .mean()
        )

        return ReaderData(
            coords=rdata.coords[max_time_idx, :].squeeze(),
            geoinfos=np.concat(
                [
                    data[self.averaging_geoinfos].values,
                    rdata.geoinfos[max_time_idx, self.non_averaging_geoinfo_idx],
                ],
                axis=1,
            ),
            data=data[channels_idx].values,
            datetimes=rdata.datetimes[max_time_idx].squeeze(),
            is_spoof=rdata.is_spoof,
        )
