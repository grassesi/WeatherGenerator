# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import torch
from numpy.typing import NDArray

import weathergen.common.config as config
import weathergen.common.io as io
from weathergen.common.io import TimeRange, zarrio_writer
from weathergen.datasets.data_reader_base import TimeWindowHandler

_logger = logging.getLogger(__name__)


def _empty_step(n_samples: int, n_ens: int, n_channels: int):
    """Zero-sized target/prediction entries for a step that carries no data."""
    return (
        [np.zeros((n_ens, 0, n_channels), dtype=np.float32) for _ in range(n_samples)],
        [np.zeros((0, n_channels), dtype=np.float32) for _ in range(n_samples)],
        [np.zeros((0, 2), dtype=np.float32) for _ in range(n_samples)],
        [np.array([]).astype("datetime64[ns]") for _ in range(n_samples)],
    )


def _to_numpy(array) -> NDArray:
    """Coordinates arrive as tensors from the target half and as arrays from the source half."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _merge_steps(parts, lens, axis):
    """Concatenate per-step arrays so that each sample's rows stay one contiguous run.

    `OutputBatchData` slices a sample out as `sum(lens[:sample])` onwards, so the merged layout
    has to be sample-major: every step of sample 0, then every step of sample 1, and so on.
    """
    offsets = [np.cumsum([0, *step_lens]) for step_lens in lens]
    runs = [
        part[(slice(None),) * axis + (slice(off[i], off[i + 1]),)]
        for i in range(len(lens[0]))
        for part, off in zip(parts, offsets, strict=True)
    ]
    return np.concatenate(runs, axis=axis)


def write_output(
    cf, val_cfg, batch_size, mini_epoch, batch_idx, dn_data, batch, model_output, target_aux_out
):
    """
    Interface for writing model output
    """

    # TODO: how to handle multiple physical loss terms
    outputs_physical = [
        loss_name
        for i, (loss_name, loss_term) in enumerate(val_cfg.losses.items())
        if loss_term.type == "LossPhysical"
    ]
    assert len(outputs_physical) == 1
    target_aux_out = target_aux_out[outputs_physical[0]]

    # collect all target / prediction-related information
    fp32 = torch.float32
    preds_all, targets_all, targets_coords_all, targets_times_all = [], [], [], []
    preds_coords_all, preds_times_all = [], []

    # _get_output_length clamps to at least one output step, so this always holds
    assert len(batch.get_output_idxs()) > 0, "Batch carries no output steps."
    forecast_offset = batch.get_output_idxs()[0]

    # the chunk describes which forecast steps it holds, including the leading empty steps
    # that the first chunk keeps so it is indexed by global forecast step
    timestep_idxs = model_output.forecast_steps

    n_samples = len(batch.get_source_samples().get_samples())
    targets_lens = []
    preds_lens = []

    for t_idx in timestep_idxs:
        preds_all += [[]]
        targets_all += [[]]
        targets_coords_all += [[]]
        targets_times_all += [[]]
        preds_coords_all += [[]]
        preds_times_all += [[]]
        targets_lens += [[]]
        preds_lens += [[]]
        for sname in cf.streams.keys():
            chunk_idx = model_output.chunk_idx(t_idx)
            assert model_output.forecast_steps[chunk_idx] == t_idx, (
                f"Prediction at index {chunk_idx} is valid for forecast step "
                f"{model_output.forecast_steps[chunk_idx]}, but the target is valid for {t_idx}."
            )

            n_channels = len(cf.streams[sname].val_target_channels)

            # leading empty steps of the first chunk carry a source but no target/prediction
            if t_idx < forecast_offset:
                preds_s, targets_s, t_coords_s, t_times_s = _empty_step(n_samples, 1, n_channels)
                preds_coords_s, preds_times_s = t_coords_s, t_times_s

            else:
                preds = model_output.get_physical_prediction(chunk_idx, sname)
                targets = target_aux_out.physical[t_idx][sname]["target"]

                preds_s, targets_s, t_coords_s, t_times_s = [], [], [], []
                preds_coords_s, preds_times_s = [], []

                # spoofed step: the target window lies outside the dataset, so the prediction
                # is written in full at the fallback geometry and the target is left empty
                is_spoof = target_aux_out.physical[t_idx][sname]["is_spoof"][0]
                if is_spoof:
                    _logger.debug(
                        f"Stream '{sname}' at t_idx={t_idx} is spoof "
                        "(target time window is outside the dataset range); "
                        "writing model predictions with empty targets."
                    )

                # handle forcing streams or if sample is empty
                if preds is None:
                    # preds are empty so create copy of target and add ensemble dimension
                    assert targets[0].shape[0] == 0, "Empty preds but non-empty targets."
                    preds = [
                        target.reshape(0, n_channels).unsqueeze(0)
                        if target.numel() == 0
                        else target.clone().unsqueeze(0)
                        for target in targets
                    ]

                for i_batch, (pred, target) in enumerate(zip(preds, targets, strict=True)):
                    target_data = target_aux_out.physical[t_idx][sname]
                    t_coords = target_data["target_coords"][i_batch]
                    t_times = target_data["target_times"][i_batch]

                    # without a target half (inference_only) there is no reordering, and rows
                    # stay in token order, which is the order of the coords they came with
                    idxs_inv = target_data["idxs_inv"][i_batch]
                    if idxs_inv is not None and (
                        not isinstance(idxs_inv, torch.Tensor) or idxs_inv.numel() > 0
                    ):
                        pred = pred[:, idxs_inv]
                        t_coords = t_coords[idxs_inv]
                        t_times = t_times[idxs_inv]
                        if not is_spoof:
                            target = target[idxs_inv]

                    # denormalize predictions and map to storage format
                    preds_s += [dn_data(sname, pred.to(fp32)).detach().cpu().numpy()]
                    preds_coords_s += [_to_numpy(t_coords)]
                    preds_times_s += [np.asarray(t_times).astype("datetime64[ns]")]

                    if is_spoof or target.numel() == 0:
                        # no ground truth for this step: write an empty target so the store
                        # signals it, while the prediction above is written in full
                        targets_s += [np.zeros((0, n_channels), dtype=np.float32)]
                        t_coords_s += [np.zeros((0, 2), dtype=np.float32)]
                        t_times_s += [np.array([], dtype="datetime64[ns]")]
                    else:
                        targets_s += [dn_data(sname, target.to(fp32)).detach().cpu().numpy()]
                        t_coords_s += [_to_numpy(t_coords)]
                        t_times_s += [np.asarray(t_times).astype("datetime64[ns]")]

            targets_lens[-1] += [[t.shape[0] for t in t_coords_s]]
            preds_lens[-1] += [[p.shape[0] for p in preds_coords_s]]

            preds_all[-1] += [np.concatenate(preds_s, axis=1)]
            targets_all[-1] += [np.concatenate(targets_s)]
            targets_coords_all[-1] += [np.concatenate(t_coords_s)]
            targets_times_all[-1] += [np.concatenate(t_times_s)]
            preds_coords_all[-1] += [np.concatenate(preds_coords_s)]
            preds_times_all[-1] += [np.concatenate(preds_times_s)]

    if len(preds_all) == 0 or np.array([p.shape[1] for pp in preds_all for p in pp]).sum() == 0:
        _logger.warning("Writing no data since predictions are empty.")
        return

    # One store group per rollout chunk, keyed chunk index + forecast_offset: a chunk's steps
    # become one forecast step whose points carry several valid times, which the export and
    # evaluation readers split back out. Step 0 keeps its own source-only group, and with
    # chunk_size unset every step is its own group, exactly as before.
    chunk_size = val_cfg.get("forecast", {}).get("chunk_size") or 1
    if chunk_size > 1:
        groups: dict[int, list[int]] = {}
        for pos, t_idx in enumerate(timestep_idxs):
            key = t_idx
            if t_idx >= forecast_offset:
                key = forecast_offset + (t_idx - forecast_offset) // chunk_size
            groups.setdefault(key, []).append(pos)

        # predictions and targets can differ in row count (a spoofed or target-less step
        # holds predictions but no target), so each is sliced by its own lens
        def _group(per_step, lens, axis):
            return [
                [
                    _merge_steps([per_step[p][s] for p in ps], [lens[p][s] for p in ps], axis)
                    for s in range(len(per_step[ps[0]]))
                ]
                for ps in groups.values()
            ]

        def _group_lens(lens):
            return [
                [
                    [sum(ls) for ls in zip(*(lens[p][s] for p in ps), strict=True)]
                    for s in range(len(lens[ps[0]]))
                ]
                for ps in groups.values()
            ]

        preds_all = _group(preds_all, preds_lens, 1)
        preds_coords_all = _group(preds_coords_all, preds_lens, 0)
        preds_times_all = _group(preds_times_all, preds_lens, 0)
        targets_all = _group(targets_all, targets_lens, 0)
        targets_coords_all = _group(targets_coords_all, targets_lens, 0)
        targets_times_all = _group(targets_times_all, targets_lens, 0)
        targets_lens = _group_lens(targets_lens)
        preds_lens = _group_lens(preds_lens)
        timestep_idxs = list(groups)

    # collect source information
    sources = []
    for sample in batch.get_source_samples().get_samples():
        sources += [[]]
        for _, stream_data in sample.streams_data.items():
            # TODO: support multiple input steps
            sources[-1] += [stream_data.source_raw[0]]

    sample_idxs = [
        list(sample.streams_data.values())[0].sample_idx
        for sample in batch.get_source_samples().get_samples()
    ]

    # more prep work

    # output stream names to be written, use specified ones or all if nothing specified
    stream_names = list(cf.streams.keys())
    stream_infos = list(cf.streams.values())
    if val_cfg.get("output").get("streams") is not None:
        output_stream_names = val_cfg.output.streams
    else:
        output_stream_names = stream_names

    output_streams = {name: stream_names.index(name) for name in output_stream_names}
    _logger.debug(f"Using output streams: {output_streams} from streams: {stream_names}")

    target_channels: list[list[str]] = [list(stream.val_target_channels) for stream in stream_infos]
    source_channels: list[list[str]] = [list(stream.val_source_channels) for stream in stream_infos]

    geoinfo_channels = [[] for _ in stream_infos]  # TODO obtain channels

    # calculate global sample indices for this batch by offsetting by sample_start
    sample_start = batch_idx * batch_size

    # write output

    start_date = val_cfg.start_date
    end_date = val_cfg.end_date

    twh = TimeWindowHandler(
        start_date,
        end_date,
        val_cfg.time_window_len,
        val_cfg.time_window_step,
    )
    source_windows = (twh.window(idx) for idx in sample_idxs)
    source_intervals = [TimeRange(window.start, window.end) for window in source_windows]

    data = io.OutputBatchData(
        sources,
        source_intervals,
        targets_all,
        preds_all,
        targets_coords_all,
        targets_times_all,
        targets_lens,
        output_streams,
        target_channels,
        source_channels,
        geoinfo_channels,
        sample_start,
        forecast_offset,
        timestep_idxs,
        preds_coords=preds_coords_all,
        preds_times=preds_times_all,
        preds_lens=preds_lens,
    )
    with zarrio_writer(config.get_path_results(cf, mini_epoch)) as zio:
        for subset in data.items():
            zio.write_zarr(subset)
