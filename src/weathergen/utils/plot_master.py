from weathergen.common.io import ZarrIO
from datetime import datetime
from functools import partial
from itertools import islice
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from weathergen.common.io import OutputItem, zarrio_reader

NPROCS = 10
CHUNK_SIZE = 1
KELVIN = 273.15
RESULTS_DIR = Path("/p/home/jusers/grasse1/juwels/WeatherGenerator/results")
RUN_ID = "ne7wcujt"


def global_temperature_mean(item: OutputItem, climatology):
    return {
        "target": np.nanmean(item.target.data),
        "prediction": np.nanmean(item.prediction.data),
        "anomaly": np.nanmean(item.prediction.data - climatology),
    }


def process_item(work_item: pd.Index, reader: ZarrIO, stream: str, index: dict[str, int]):
    kwargs = {
        "stream": stream,
        work_item.name: work_item.values[0],
        **index,
    }

    item = reader.get_data(**kwargs)
    result = pd.DataFrame(
        data={
            "source_interval_start": item.target.source_interval.start,
            "source_interval_end": item.target.source_interval.end,
            "channel": item.target.channels[0],
            **global_temperature_mean(item, climatology),
            **kwargs,
        },
        index=work_item,
    )

    return result


def plot_means_lead_time(result_path, sample, stream, n_steps=None):
    with zarrio_reader(result_path) as reader, Pool(NPROCS) as p:
        worker_fun = partial(process_item, reader=reader, sample=sample, stream=stream)
        fsteps = list(sorted(int(fstep) for fstep in reader.forecast_steps))[:n_steps]
        return pd.concat(p.imap(worker_fun, fsteps, chunksize=max(len(fsteps) // NPROCS, 1)))


def get_timeseries(result_path: Path, stream: str, fstep=None, sample=None):
    assert (fstep is None) != (sample is None)
    with zarrio_reader(result_path) as reader:
        if fstep is not None:
            index = {"forecast_step": fstep}
            aggregation_axis = reader.samples
            dim = "sample"
        elif sample is not None:
            index = {"sample": sample}
            aggregation_axis = reader.forecast_steps
            dim = "forecast_step"
        else:
            raise ValueError("foo")

        aggregation_axis_sorted = sorted(int(item) for item in aggregation_axis)
        work_items = [pd.Index([item], name=dim) for item in aggregation_axis_sorted]
        worker_fun = partial(process_item, reader=reader, index=index, stream=stream)

        with Pool(NPROCS) as p:
            chunk_size = max(len(work_items) // NPROCS, 1)
            return pd.concat(p.imap(worker_fun, work_items, chunksize=chunk_size))


def main():
    start = datetime.now()
    file_name = "validation_chkpt00000_rank0000.zip"
    result_path = RESULTS_DIR / RUN_ID / file_name
    print(plot_means_lead_time(result_path, 0, "ERA5-Ocean"))
    print(f"total time :{datetime.now() - start}")


if __name__ == "__main__":
    main()
