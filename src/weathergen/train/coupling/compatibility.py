# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Setup checks that the two checkpoints a coupled run pairs can actually be paired."""

from __future__ import annotations

import logging
import typing

import numpy as np
import torch

import weathergen.common.config as config
from weathergen.common.config import timedelta_to_str
from weathergen.datasets.averaging import AveragingReader
from weathergen.datasets.data_reader_base import DataReaderBase, WrappedDataReader
from weathergen.datasets.upsampling import UpsamplingReader
from weathergen.train.coupling.checks import emission_cadence
from weathergen.train.coupling.spec import produced_streams

if typing.TYPE_CHECKING:
    from weathergen.train.coupling.coupler import Coupler
    from weathergen.train.coupling.run import ModelCheckpoint

logger = logging.getLogger(__name__)

_ZERO = np.timedelta64(0, "ms")

# `ForcingEngine.init_weights_final` draws every block weight from normal(0, 0.001), so a
# freshly built engine is a near-identity by construction (`forcing.py`). An engine still
# sitting at that spread never learned anything, which is what generation 01's xcpk26es and
# j5h3is35 turned out to be.
_FFE_INIT_STD = 0.001
# Relative distance from _FFE_INIT_STD within which an engine is called untrained.
_FFE_INIT_TOL = 0.25


def _unwrap_model(model):
    """The ForcedModel underneath any DDP wrapper.

    `init_model_and_shard` wraps in DistributedDataParallel when running without FSDP, while
    `fully_shard` mutates in place and leaves the module itself reachable.
    """
    return getattr(model, "module", model)


def _reader_stack(reader: DataReaderBase) -> list[DataReaderBase]:
    """Every reader in a wrapper stack, outermost first."""
    stack = [reader]
    while isinstance(stack[-1], WrappedDataReader):
        stack.append(stack[-1]._wrapped_reader)
    return stack


def _ffe_param_count(engine) -> int:
    """Parameters in a forcing engine's blocks, counted regardless of requires_grad.

    `get_num_parameters` filters on requires_grad, which an inference run may have switched
    off, and a trained engine reading as empty would fail the identity check below for the
    wrong reason.
    """
    if engine is None:
        return 0
    return sum(p.numel() for p in engine.blocks.parameters())


def _ffe_weight_spread(engine) -> float | None:
    """Standard deviation over every forcing-engine block parameter, or None if unmeasurable.

    Under FSDP the parameters are DTensors whose `.std()` would be a collective, and a check
    that can deadlock is worse than a check that abstains. `to_local()` keeps it rank-local.
    """
    values = []
    for param in engine.blocks.parameters():
        tensor = param.detach()
        to_local = getattr(tensor, "to_local", None)
        if to_local is not None:
            tensor = to_local()
        values.append(tensor.reshape(-1).float())
    if not values:
        return None
    try:
        return float(torch.cat(values).std().item())
    except RuntimeError:
        return None


def _stream_files(reader: DataReaderBase) -> list[str]:
    """The files a reader's stream resolves to, as its stream_info records them."""
    info = getattr(reader, "stream_info", None) or {}
    return sorted(str(f) for f in (info.get("filenames") or []))


def check_forcing_engines(coupler: Coupler) -> None:
    """A component receiving a coupled forcing must have a forcing engine with weights.

    `ffe_num_blocks` defaults to 0 and a zero-block engine is the identity: no parameters,
    an empty state dict, and `ForcedModel.forward` passes the latent through untouched, so
    the forcing tokens are discarded. That is a legitimate configuration for an unforced
    component (`open_actions.md` item 1) and it is exactly what a *forced* component looks
    like when its engine failed to load -- `load_state_dict` runs with `strict=False`, so
    unmatched keys are a warning and nothing else.

    Under coupling the forcing engine is the path the exchanged field travels. An identity
    engine there transports nothing while every provenance count still reports a live
    exchange, which is the `announce_couplings` failure one layer down, in the weights
    instead of the wiring.
    """

    for coupling in coupler.live_couplings():
        consumer = coupling.consumer
        where = (
            f"Component {consumer!r} is forced by {coupling.producer!r} on "
            f"'{coupling.stream}'"
        )
        model = _unwrap_model(coupler.trainer(consumer).model)
        engine = getattr(model, "forcing_engine", None)
        num_params = _ffe_param_count(engine)

        if num_params == 0:
            msg = (
                f"{where}, but its forcing engine has no parameters "
                f"(ffe_num_blocks={coupler.config(consumer).get('ffe_num_blocks', 0)}). A "
                "zero-block engine is the identity, so the forcing tokens are built, "
                "gathered, reported as exchanged -- and then discarded. Use a checkpoint "
                "trained with ffe_num_blocks > 0 as the consumer."
            )
            raise ValueError(msg)

        std = _ffe_weight_spread(engine)
        if std is None:
            logger.info(
                f"{where}: forcing engine has {num_params} parameters; weight spread not "
                "measurable on this sharding."
            )
            continue

        logger.info(
            f"{where}: forcing engine has {num_params} parameters, weight std {std:.3e}."
        )
        if abs(std - _FFE_INIT_STD) <= _FFE_INIT_TOL * _FFE_INIT_STD:
            logger.error(
                f"{where}, but its forcing engine's weight std {std:.3e} is within "
                f"{_FFE_INIT_TOL:.0%} of the {_FFE_INIT_STD} it is initialized to, so it "
                "looks untrained and the coupling will transport almost nothing. "
                "Generation 01's xcpk26es and j5h3is35 failed exactly this way. Check "
                "that the checkpoint really finetuned the forcing engine before trusting "
                "this run."
            )


def report_checkpoint_provenance(
    coupler: Coupler, checkpoints: dict[str, ModelCheckpoint]
) -> None:
    """One line per component recording which weights this run actually paired.

    Continuation pairs a checkpoint with itself, so its provenance is the run_id. A coupled
    run pairs two, and which two is the experiment -- but nothing in the artifacts records
    it, which is what the run registry keeps having to reconstruct after the fact. No
    policy here, just the record.
    """

    for name in coupler.names:
        checkpoint = checkpoints.get(name)
        ccf = coupler.config(name)
        model = _unwrap_model(coupler.trainer(name).model)
        engine = getattr(model, "forcing_engine", None)
        ffe_params = _ffe_param_count(engine)

        size = mtime = "unknown"
        if checkpoint is not None:
            # A record that can abort the run it is recording is worse than an incomplete
            # one: resolving the model path goes through the private config, which is not
            # reachable everywhere this runs.
            try:
                path = (
                    config.get_path_model(run_id=checkpoint.run_id)
                    / f"{checkpoint.run_id}_chkpt{checkpoint.mini_epoch:05d}.chkpt"
                )
                stat = path.stat()
            except (OSError, AssertionError, ValueError, KeyError):
                pass
            else:
                size = f"{stat.st_size / 1024**3:.2f} GiB"
                mtime = str(np.datetime64(int(stat.st_mtime), "s"))

        logger.info(
            f"Component {name!r} provenance: "
            f"checkpoint={checkpoint.run_id if checkpoint else '?'}"
            f"@{checkpoint.mini_epoch if checkpoint else '?'}, "
            f"size={size}, mtime={mtime}, "
            f"healpix_level={ccf.get('healpix_level')}, "
            f"ffe_num_blocks={ccf.get('ffe_num_blocks', 0)}, ffe_params={ffe_params}, "
            f"produces={produced_streams(coupler.couplings, name) or None}."
        )


def check_exchange_grid(coupler: Coupler) -> None:
    """The two sides of a live coupling must mean the same spatio-temporal grid.

    Only the grid. `token_size` and `healpix_level` are how each component chops its own
    input up, not what the exchanged points are, and the consumer re-tokenizes what it
    receives with its own tokenizer either way -- so they are recorded by
    `report_checkpoint_provenance` and left alone here.

    **Space.** The producer emits predictions on the point set of its own reader for the
    stream, and the consumer tokenizes them as if they had come off its own disk reader.
    Two readers resolving to different files are two different point sets, and nothing
    downstream would say so.

    **Time.** A gap in temporal resolution is legitimate -- it is what the averaging and
    upsampling wrappers exist for -- but only in the direction each of them bridges.
    `DataReaderCoupling` gathers every source window whose start falls in the request, so
    a producer *finer* than the stream's native period hands the consumer more rows per
    window than it ever trained on, and only an `AveragingReader` above the coupling
    reader reduces them back. The comparison is producer cadence against the period the
    consumer's disk reader declares, because the request window length is the same in
    training and under coupling: their ratio is the factor by which the row count moved.
    This is not `coupling_reader_placement.md` P6's withdrawn criterion, which compared
    periods to *choose* a wrapper to insert; here the stack is fixed by the checkpoint and
    the comparison only asks whether it still fits (P8). A producer *coarser* than the
    native period empties the gather, and the coupling reader takes the covering window
    and restamps it, which is a zero-order hold the consumer did not necessarily train
    through.
    """

    for coupling in coupler.live_couplings():
        producer, consumer, stream = coupling.producer, coupling.consumer, coupling.stream
        where = f"Coupling {coupling.name!r} ({producer!r} -> {consumer!r}, '{stream}')"

        producer_reader = coupler.producer_reader(producer, stream)
        consumer_readers = coupler.pristine_forcings(consumer).forcing_streams[stream]
        stack = _reader_stack(consumer_readers[0])
        consumer_reader = stack[-1]

        # -- space: the same underlying dataset, hence the same points
        producer_files = _stream_files(producer_reader)
        consumer_files = _stream_files(consumer_reader)
        if producer_files != consumer_files:
            msg = (
                f"{where} exchanges a stream the two components read from different "
                f"files: {producer!r} has {producer_files}, {consumer!r} has "
                f"{consumer_files}. The producer emits on its own point set and the "
                "consumer tokenizes the result as its own, so the forcing would land on "
                "a grid the consumer never trained on."
            )
            raise ValueError(msg)

        # -- time: the producer's cadence against the period the consumer trained on
        cadence = emission_cadence(coupler.config(producer))
        period = getattr(consumer_reader, "period", None)
        if period is None or cadence == _ZERO:
            logger.info(
                f"{where}: grid check covered files only; "
                f"{consumer!r}'s reader for '{stream}' declares no period."
            )
            continue

        bridges = [
            type(r).__name__
            for r in stack
            if isinstance(r, AveragingReader | UpsamplingReader)
        ]

        if period == cadence:
            logger.info(
                f"{where}: grid agrees, producer cadence "
                f"{timedelta_to_str(cadence)} = stream period "
                f"{timedelta_to_str(period)}"
                + (f", bridged by {bridges}." if bridges else ".")
            )
        elif cadence < period:
            if not any(b == "AveragingReader" for b in bridges):
                msg = (
                    f"{where}: {producer!r} emits every {timedelta_to_str(cadence)}, "
                    f"finer than the {timedelta_to_str(period)} period {consumer!r} reads "
                    f"'{stream}' at, so every request window gathers "
                    f"{period // cadence}x the rows it did in training. The stream carries "
                    f"no AveragingReader to reduce them ({bridges or 'no wrappers'}), so "
                    "the tokenizer would see a row count the model never trained on."
                )
                raise ValueError(msg)
            logger.info(
                f"{where}: producer cadence {timedelta_to_str(cadence)} is finer than the "
                f"{timedelta_to_str(period)} stream period; AveragingReader bridges the "
                f"{period // cadence}x increase in rows per window."
            )
        else:
            logger.warning(
                f"{where}: {producer!r} emits every {timedelta_to_str(cadence)}, coarser "
                f"than the {timedelta_to_str(period)} period {consumer!r} reads '{stream}' "
                "at. The coupling reader serves the covering window restamped, so the "
                f"consumer sees a field up to {timedelta_to_str(cadence - period)} stale "
                "where training varied every window"
                + (f" ({bridges} also in the stack)." if bridges else ".")
            )


def check_exchange_masks(coupler: Coupler) -> None:
    """An exchanged channel that is NaN on disk must be masked in the producer's prediction.

    `_prime` hands the consumer the producer's ground truth for chunk 0, NaN wherever the
    data is (SST over land), and the consumer's tokenizer turns that NaN into the same
    `mask_value` it saw in training. Every later chunk is a prediction, finite everywhere
    unless the producer masks it via `streams.<stream>.mask_predictions`. Unmasked, the
    consumer is forced from chunk 1 on by land values it never saw, with nothing to show
    for it (`optional_target_sst_masking.md` C, M3).

    The producer's disk reader knows which channels carry NaNs (`nan_channels`); None
    means it cannot say, and then the check abstains rather than guesses.
    """

    for coupling in coupler.live_couplings():
        producer, consumer, stream = coupling.producer, coupling.consumer, coupling.stream
        where = f"Coupling {coupling.name!r} ({producer!r} -> {consumer!r}, '{stream}')"

        # what crosses is what the coupling reader maps: the consumer's source channels,
        # taken by name from the producer's target channels
        producer_reader = coupler.producer_reader(producer, stream)
        consumer_readers = coupler.pristine_forcings(consumer).forcing_streams[stream]
        needed = set(_reader_stack(consumer_readers[0])[-1].source_channels)
        exchanged = [ch for ch in producer_reader.target_channels if ch in needed]

        stream_cfg = coupler.config(producer).streams[stream]
        mask = list(stream_cfg.get("mask_predictions", None) or [])

        readers = coupler.trainer(producer).dataset.streams_datasets[stream].readers
        nans = [getattr(_reader_stack(r)[-1], "nan_channels", None) for r in readers]
        if any(nan is None for nan in nans):
            logger.warning(
                f"{where}: {producer!r}'s reader cannot say which channels carry NaNs, so "
                f"whether the exchanged channels {exchanged} need masking was not checked. "
                f"Masked: {[ch for ch in exchanged if ch in mask]}."
            )
            continue

        nan = frozenset().union(*nans)
        unmasked = [ch for ch in exchanged if ch in nan and ch not in mask]
        if unmasked:
            msg = (
                f"{where}: {producer!r} hands over {unmasked}, which are NaN on disk, but "
                f"its stream config does not mask them (mask_predictions={mask}). Chunk 0 "
                f"reaches {consumer!r} with NaN there and every later chunk with finite "
                "predictions the consumer never trained on. Pass "
                f"--options '{producer}:streams.{stream}.mask_predictions="
                f"[{','.join(sorted(set(mask) | set(unmasked)))}]'."
            )
            raise ValueError(msg)

        logger.info(
            f"{where}: exchanged channels {exchanged}, masked "
            f"{[ch for ch in exchanged if ch in mask]}, NaN on disk {sorted(nan)}."
        )
