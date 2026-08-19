#!/usr/bin/env python3
# (C) Copyright 2026 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Build a climatology store from an anemoi zarr dataset.

The output is a drop-in sibling of the climatology stores already in
``<data_path_aux>/climatology/``, which ``packages/evaluate`` resolves by name (see
``weathergen.evaluate.io.wegen_reader.WeatherGenReader.get_climatology_filename``) and
reads through ``weathergen.evaluate.utils.clim_utils``.

Layout, taken from the existing ERA5 stores::

    data  (time, statistic, channels, grid_points)  float32, one chunk per timestep
    time       365 * steps_per_day labels, ``2000-01-01T00 + i * frequency``
    statistic  ['mean', 'q20', 'q33', 'q40', 'q60', 'q67', 'q80']
    channels   the source dataset's variable names
    latitude / longitude  over grid_points

Slot ``i`` holds the statistic over the reference years, sampling each year at the same
(month, day, hour) as the slot's year-2000 label. The labels live in a leap year, so the
Feb-29 slot takes **Mar 1** from non-leap years -- this reproduces the existing ERA5 store
exactly (verified to float32 rounding on every statistic).

Quantiles are computed by sorting along the year axis and interpolating linearly, which is
~150x faster than ``np.nanquantile`` and exactly equivalent as long as every grid point is
either NaN in all reference years or in none. That holds for the ERA5/s2s land masks; the
script checks it per slot and falls back to the nan-aware path if it ever fails.

Example
-------
::

    python compute_climatology.py \\
        --source .../aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v2-s2s-predictors.zarr \\
        --out    .../<same name>_climatology.zarr \\
        --years 1980 2020
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import zarr
from numcodecs import Blosc

_logger = logging.getLogger("compute_climatology")

STATISTICS = ["mean", "q20", "q33", "q40", "q60", "q67", "q80"]
QUANTILES = {"q20": 0.2, "q33": 0.33, "q40": 0.4, "q60": 0.6, "q67": 0.67, "q80": 0.8}

# The slot labels live in this dummy year. It is a leap year, which is why the Feb-29 slot
# exists at all and why the label sequence stops at Dec 30 after 365 days.
LABEL_YEAR = 2000
DAYS_PER_YEAR = 365


def slot_labels(steps_per_day: int) -> pd.DatetimeIndex:
    """The year-2000 timestamps that label each climatology slot."""
    return pd.date_range(
        f"{LABEL_YEAR}-01-01",
        periods=DAYS_PER_YEAR * steps_per_day,
        freq=pd.Timedelta(days=1) / steps_per_day,
    )


def build_index_table(
    dates: pd.DatetimeIndex, labels: pd.DatetimeIndex, years: list[int]
) -> npt.NDArray:
    """Map each (slot, reference year) pair to an index into the source dataset.

    Returns an ``(n_slots, n_years)`` array. Raises if any pair is unresolvable, so a
    truncated or irregular source is caught before hours of work are spent on it.
    """
    position = {(d.year, d.month, d.day, d.hour): i for i, d in enumerate(dates)}

    table = np.full((len(labels), len(years)), -1, dtype=np.int64)
    for slot, label in enumerate(labels):
        for j, year in enumerate(years):
            key = (year, label.month, label.day, label.hour)
            if key not in position and (label.month, label.day) == (2, 29):
                # non-leap reference year: the Feb-29 slot takes Mar 1 at the same hour
                key = (year, 3, 1, label.hour)
            table[slot, j] = position.get(key, -1)

    missing = np.argwhere(table < 0)
    if missing.size:
        slot, j = missing[0]
        raise ValueError(
            f"{len(missing)} (slot, year) pairs are absent from the source dataset, "
            f"first is slot {slot} ({labels[slot]}) in year {years[j]}."
        )
    return table


def slot_statistics(block: npt.NDArray) -> npt.NDArray:
    """Mean and quantiles over axis 0 of ``block``.

    ``block`` is ``(n_years, n_channels, n_grid_points)``; the result is
    ``(len(STATISTICS), n_channels, n_grid_points)``.
    """
    n_years = block.shape[0]
    ordered = np.sort(block, axis=0)  # NaNs sort to the end

    # A point is safe if it is NaN in every reference year (all-NaN, which correctly
    # yields NaN) or in none of them. A point that is NaN in only some years would be
    # silently mis-ranked by the sort, so such a slot takes the slow but correct path.
    mixed = ~np.isnan(ordered[0]) & np.isnan(ordered[-1])
    out = np.empty((len(STATISTICS), *block.shape[1:]), dtype=np.float32)

    if mixed.any():
        _logger.warning(
            "%d point(s) are NaN in some but not all reference years;"
            " falling back to nan-aware statistics for this slot.",
            int(mixed.sum()),
        )
        out[0] = np.nanmean(block, axis=0)
        out[1:] = np.nanquantile(block, [QUANTILES[s] for s in STATISTICS[1:]], axis=0)
        return out

    out[0] = block.mean(axis=0)
    for k, name in enumerate(STATISTICS[1:], start=1):
        position = QUANTILES[name] * (n_years - 1)
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        weight = np.float32(position - lower)
        out[k] = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return out


def create_store(
    path: Path,
    labels: pd.DatetimeIndex,
    channels: list[str],
    latitudes: npt.NDArray,
    longitudes: npt.NDArray,
    attrs: dict,
) -> zarr.Group:
    """Create the empty store with the same schema as the existing ERA5 climatologies."""
    n_grid = latitudes.size
    shape = (len(labels), len(STATISTICS), len(channels), n_grid)
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)

    root = zarr.open_group(str(path), mode="w", zarr_format=2)

    data = root.create_array(
        "data",
        shape=shape,
        chunks=(1, len(STATISTICS), len(channels), n_grid),
        dtype="float32",
        fill_value=np.nan,
        compressors=[compressor],
    )
    data.attrs["_ARRAY_DIMENSIONS"] = ["time", "statistic", "channels", "grid_points"]
    data.attrs["coordinates"] = "latitude longitude"

    # Coordinates must not carry a fill value that any real element could equal: xarray
    # reads the zarr fill value as _FillValue and masks matching elements. zarr's default
    # is 0, which would turn hour 0 into NaT and longitude 0.0 into NaN.
    hours = (labels - pd.Timestamp(f"{LABEL_YEAR}-01-01")) // pd.Timedelta(hours=1)
    time = root.create_array(
        "time",
        shape=(len(labels),),
        chunks=(len(labels),),
        dtype="int64",
        fill_value=None,
        compressors=[compressor],
    )
    time[:] = np.asarray(hours, dtype="int64")
    time.attrs["_ARRAY_DIMENSIONS"] = ["time"]
    time.attrs["units"] = f"hours since {LABEL_YEAR}-01-01 00:00:00"
    time.attrs["calendar"] = "proleptic_gregorian"

    for name, values, dim, fill in (
        ("statistic", np.array(STATISTICS), "statistic", None),
        ("channels", np.array(channels), "channels", None),
        ("latitude", latitudes, "grid_points", np.nan),
        ("longitude", longitudes, "grid_points", np.nan),
    ):
        array = root.create_array(
            name,
            shape=values.shape,
            chunks=values.shape,
            dtype=values.dtype,
            fill_value=fill,
            compressors=[compressor],
        )
        array[:] = values
        array.attrs["_ARRAY_DIMENSIONS"] = [dim]

    root.attrs.update(attrs)
    return root


def chunk_written(path: Path, slot: int) -> bool:
    """True if slot ``slot`` already has its chunk on disk (one chunk per timestep)."""
    return (path / "data" / f"{slot}.0.0.0").exists()


def read_source(path: Path) -> dict:
    """Pull the pieces of an anemoi zarr dataset this script needs."""
    source = zarr.open(str(path), mode="r")
    attrs = dict(source.attrs)

    def as_list(value):
        return json.loads(value) if isinstance(value, str) else list(value)

    return {
        "data": source["data"],
        "dates": pd.DatetimeIndex(np.asarray(source["dates"][:]).astype("datetime64[s]")),
        "latitudes": np.asarray(source["latitudes"][:]),
        "longitudes": np.asarray(source["longitudes"][:]),
        "channels": as_list(attrs["variables"]),
        "variables_with_nans": as_list(attrs.get("variables_with_nans", [])),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a climatology store from an anemoi zarr dataset."
    )
    parser.add_argument("--source", type=Path, required=True, help="anemoi zarr dataset")
    parser.add_argument("--out", type=Path, required=True, help="climatology zarr to write")
    parser.add_argument(
        "--years",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=(1980, 2020),
        help="inclusive reference period (default: 1980 2020, as for the ERA5 stores)",
    )
    parser.add_argument(
        "--slots",
        default=None,
        metavar="START:END",
        help="half-open slot range to fill, for splitting the run across array tasks",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep an existing store and skip slots whose chunk is already on disk",
    )
    parser.add_argument("--description", default="", help="free-text description attribute")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )

    source = read_source(args.source)
    dates = source["dates"]
    channels = source["channels"]

    frequency = pd.Timedelta(dates[1] - dates[0])
    steps_per_day = int(pd.Timedelta(days=1) / frequency)
    labels = slot_labels(steps_per_day)
    years = list(range(args.years[0], args.years[1] + 1))

    _logger.info(
        "source %s: %d dates %s..%s, %d channels, %d grid points, %s frequency",
        args.source.name,
        len(dates),
        dates[0].date(),
        dates[-1].date(),
        len(channels),
        source["latitudes"].size,
        frequency,
    )
    _logger.info(
        "building %d slots x %d statistics x %d channels over %d years (%d..%d)",
        len(labels),
        len(STATISTICS),
        len(channels),
        len(years),
        years[0],
        years[-1],
    )

    table = build_index_table(dates, labels, years)

    attrs = {
        "chunking_strategy": "1 timestep × all statistics × all channels × all grid_points",
        "creation_date": dt.datetime.now().isoformat(),
        "data_structure": "time × statistic × channels × grid_points",
        "description": args.description
        or f"Climatology of {args.source.name} over {years[0]}-{years[-1]}.",
        "dtype": "float32",
        "leap_year_handling": "Feb 29 mapped to Mar 1 in non-leap years",
        "processing_method": "multi-year mean and quantiles",
        "source_dataset": str(args.source),
        "source_years_count": len(years),
        "source_years_end": years[-1],
        "source_years_start": years[0],
        "timesteps_per_year": len(labels),
        "variables_with_nans": source["variables_with_nans"],
    }

    if args.resume and args.out.exists():
        root = zarr.open_group(str(args.out), mode="r+", zarr_format=2)
        _logger.info("resuming into existing store %s", args.out)
    else:
        root = create_store(
            args.out, labels, channels, source["latitudes"], source["longitudes"], attrs
        )
        _logger.info("created %s", args.out)

    out = root["data"]
    data = source["data"]

    start, end = 0, len(labels)
    if args.slots:
        start, end = (int(part) for part in args.slots.split(":"))

    written = 0
    for slot in range(start, end):
        if args.resume and chunk_written(args.out, slot):
            continue
        block = np.stack([data[int(i), :, 0, :] for i in table[slot]])
        out[slot] = slot_statistics(block)
        written += 1
        if written % 50 == 0:
            _logger.info("slot %d of [%d, %d) (%s)", slot, start, end, labels[slot])

    _logger.info("wrote %d slots in [%d, %d)", written, start, end)

    zarr.consolidate_metadata(root.store)
    _logger.info("consolidated metadata; done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
