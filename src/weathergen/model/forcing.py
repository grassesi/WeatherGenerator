# ruff: noqa: T201

from __future__ import annotations

import dataclasses
import itertools as it
import logging
from typing import Any

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config, timedelta_to_str
from weathergen.common.io import IOReaderData
from weathergen.datasets.batch import BatchSamples, SampleMetaData
from weathergen.datasets.coupling_reader import (
    DataReaderCoupling,
    ForcingProvenance,
    forcing_lag,
)
from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    DTRange,
    TimeWindowHandler,
    WrappedDataReader,
    rebase_innermost,
    shifted,
)
from weathergen.datasets.masking import MaskData
from weathergen.datasets.stream_data import StreamData, spoof
from weathergen.datasets.tokenizer_masking import TokenizerMasking
from weathergen.datasets.utils import get_tokens_lens
from weathergen.model.attention import MultiCrossAttentionHeadVarlen
from weathergen.model.chunking import ChunkInfo
from weathergen.model.layers import MLP
from weathergen.model.model import Model, ModelOutput, ModelParams
from weathergen.model.utils import get_num_parameters
from weathergen.utils.utils import get_dtype

_logger = logging.getLogger(__name__)


def _innermost(reader: DataReaderBase) -> DataReaderBase:
    """Walk a wrapper stack down to the reader that actually holds the data."""
    while isinstance(reader, WrappedDataReader):
        reader = reader._wrapped_reader
    return reader


class ForcedModel(Model):
    def __init__(self, cf: Config, sources_size, targets_num_channels, targets_coords_size):
        super().__init__(cf, sources_size, targets_num_channels, targets_coords_size)

        self.forcing_engine = ForcingEngine(self.cf, self.cf.get("ffe_num_blocks", 0))

    def _gather_parameters(self) -> dict[str, int]:
        num_params = super()._gather_parameters()
        num_params["ffe"] = get_num_parameters(self.forcing_engine.blocks)
        return num_params

    def _print_components(self, num_params):
        super()._print_components(num_params)
        print(f" Forecast forcing engine: {num_params['ffe']:,}")

    def forward(
        self,
        model_params: ModelParams,
        input: BatchSamples | ModelOutput,
        chunk: ChunkInfo,
        dynamic_forcings: ForcingInput,
    ) -> ModelOutput:
        """Forward pass of the model

        Tokens are processed through the model components, which were defined in the create method.
        Args:
            model_params : Query and embedding parameters
            input : the batch's source samples, or the previous chunk's output
            chunk : the tile of the rollout to advance, i.e. its global forecast steps
            dynamic_forcings : forcing sources sampled per forecast step
        Returns:
            A list containing all prediction results
        """
        source_samples, tokens, posteriors = self._get_initial_conditions(input, model_params)

        source_masks, source_sampling_idxs = self._get_source_masks_sample_idxs(source_samples)

        forecast_steps = chunk.steps

        output = ModelOutput(chunk, source_samples)
        # posteriors come from encoding the source window, so they exist only on the first chunk
        if posteriors is not None:
            output.add_latent_prediction(0, "posteriors", posteriors)

        # Allow for pushforward trick TODO: enable
        # roll-out in latent space, iterate and generate output over requested output steps
        p_fwd = self.cf.training_config.get("forecast", {}).get("pushforward", False)
        final_step = source_samples.get_output_idxs()[-1]
        for step in forecast_steps:
            # The window a step predicts, on this component's own timeline. The lag is not
            # encoded here: each forcing stream is read on a handler already shifted by its own
            # `forcing_lag`, so one expression serves every stream and every scheme. Multiplying
            # by the stride is also the minimal correct form -- without it the lag grew linearly
            # with the forecast step whenever `forecast.time_step > time_window_step`.
            forcing_idx = step * chunk.step_stride

            if not dynamic_forcings.is_empty:
                # reembedd forcings
                forcing_sampling_idxs = [
                    sample_idx + forcing_idx for sample_idx in source_sampling_idxs
                ]
                forcings = dynamic_forcings.get_data(forcing_sampling_idxs, source_masks)
                forcings = forcings.to_device(tokens.device)
                forcing_tokens, _ = self.encoder(model_params, forcings)

                # combine forcings with current latent space
                tokens = self.forcing_engine(tokens, forcing_tokens)

            without_grad = p_fwd and self.training and step != final_step
            if without_grad:
                # Pushforward mode: advance tokens without grad; no decoding
                with torch.no_grad():
                    tokens = self.forecast_engine(tokens, step, coords=model_params.rope_coords)
                continue

            tokens = self.forecast_engine(tokens, step, coords=model_params.rope_coords)
            # decoder predictions
            output = self.predict_decoders(model_params, step, tokens, source_samples, output)
            # latent predictions (raw and with SSL heads)
            output = self.predict_latent(model_params, step, tokens, source_samples, output)

        return output

    def _get_source_masks_sample_idxs(
        self, source_samples: BatchSamples
    ) -> tuple[list[dict[str, Any]], list[int]]:
        source_masks = [
            {stream: meta_info.mask for stream, meta_info in sample.meta_info.items()}
            for sample in source_samples.samples
        ]

        source_sampling_idxs = []
        for sample in source_samples.samples:
            sample_idxs = {
                stream_data.sample_idx
                for stream_data in sample.streams_data.values()
                if stream_data is not None
            }
            assert len(sample_idxs) == 1, (
                f"Expected exactly one sampling index per sample, got {sorted(sample_idxs)}."
            )
            source_sampling_idxs.append(next(iter(sample_idxs)))

        return source_masks, source_sampling_idxs


class ForcingInput:
    """The dynamic forcing streams of one component, and the timelines they are read on.

    Every stream is read through a `DataReaderCoupling` -- with `is_forced=False` here, where
    the rows come from the component's own dataset, and with a producer bound to it when a
    `Coupler` later substitutes a partner's predictions. The two paths then differ in where the
    numbers come from and in nothing else: the same lag, the same window arithmetic, the same
    reduction by the stream's own wrappers (`forcing_lag_design.md` L7).
    """

    def __init__(
        self,
        stage: str,
        time_window_handler: TimeWindowHandler,
        forcing_streams: dict[str, list[DataReaderBase]],
        tokenizer: TokenizerMasking,
        healpix_level: int,
        forecast_offset: int = 1,
    ):
        self.stage = stage
        self.forcing_window_len = 1
        # the component's own data level: the tokenizer bins the forcing into these cells, and a
        # spoofed window has to land on the same grid the model was trained on
        self.healpix_lvl = int(healpix_level)
        tokenizer_level = getattr(tokenizer, "healpix_level", self.healpix_lvl)
        if tokenizer_level != self.healpix_lvl:
            msg = (
                f"ForcingInput built at healpix_level {self.healpix_lvl}, but its tokenizer bins "
                f"at {tokenizer_level}; the forcing cells would not line up with the tokens."
            )
            raise ValueError(msg)
        self.num_healpix_cells = 12 * 4**self.healpix_lvl

        self.time_window_handler = time_window_handler
        self.tokenizer = tokenizer
        self.tokenize_spacetime = True  # TODO hardcoded, do properly
        self.forecast_offset = int(forecast_offset)

        # Per stream, the timeline its requests are resolved on: this component's own handler
        # shifted earlier by the stream's forcing lag. The tokenizer stamps the window the data
        # actually came from, so it reads the same handler (L5).
        self.lags: dict[str, np.timedelta64] = {}
        self.stream_handlers: dict[str, TimeWindowHandler] = {}
        self.provenance: dict[str, ForcingProvenance] = {}
        self.forcing_streams = self._lag_streams(forcing_streams)

    def _lag_streams(
        self, forcing_streams: dict[str, list[DataReaderBase]]
    ) -> dict[str, list[DataReaderBase]]:
        """Read every stream through a coupling reader on its own lagged timeline.

        A stream whose innermost reader has no sampling period -- an observation stream -- is
        left on its own reader: the gathering a lag is resolved by needs a grid. Configuring a
        lag on one is rejected rather than silently ignored.
        """

        lagged: dict[str, list[DataReaderBase]] = {}
        window_step = self.time_window_handler.t_window_step

        for stream, readers in forcing_streams.items():
            stream_info = readers[0].stream_info if readers else {}
            lag = forcing_lag(stream_info, window_step, self.forecast_offset)
            default = np.timedelta64(self.forecast_offset * window_step, "ms")

            self.lags[stream] = lag
            self.stream_handlers[stream] = shifted(self.time_window_handler, lag)
            self.provenance[stream] = ForcingProvenance(stream=stream)

            periodic = [r for r in readers if getattr(_innermost(r), "period", None) is not None]
            if len(periodic) != len(readers):
                if lag != default:
                    msg = (
                        f"Stream '{stream}' sets forcing_lag {timedelta_to_str(lag)} but is not "
                        "read from a gridded reader, so the window its rows would be gathered "
                        "from is undefined."
                    )
                    raise ValueError(msg)
                lagged[stream] = readers
                continue

            lagged[stream] = [
                rebase_innermost(
                    reader,
                    lambda base, _s=stream: DataReaderCoupling(
                        base,
                        _s,
                        request_handler=self.stream_handlers[_s],
                        is_forced=False,
                        provenance=self.provenance[_s],
                    ),
                )
                for reader in readers
            ]

        for stream, lag in self.lags.items():
            _logger.info(
                f"Forcing stream '{stream}' is sampled {timedelta_to_str(lag)} before the "
                "window it forces."
            )

        return lagged

    @property
    def is_empty(self) -> bool:
        return len(self.forcing_streams) == 0

    def handler(self, stream: str) -> TimeWindowHandler:
        """The lagged timeline `stream` is read on, falling back to the unlagged one."""
        return self.stream_handlers.get(stream, self.time_window_handler)

    def get_data(
        self, sampling_idxs: list[int], meta_infos: list[dict[str, torch.Tensor]]
    ) -> BatchSamples:
        """
        Sample all forcing sources for the input window corresponding to a rollout step.

        Args:
          sampling_idxs: Dataset indices to retrieve forcing sources for, one per batch sample.

        Returns: Data that can be ingested by the Encoder.
        """

        samples = range(len(meta_infos))
        assert len(sampling_idxs) == len(meta_infos), (
            "Expected one forcing sampling index per sample, "
            f"got {len(sampling_idxs)} indices for {len(meta_infos)} samples."
        )

        forcing_samples = BatchSamples(
            stream_names=list(self.forcing_streams.keys()),
            num_samples=len(samples),
            output_steps=1,
            output_idxs=None,  # not needed, since not used in encoder
        )

        for stream, sample in it.product(self.forcing_streams.keys(), samples):
            mask = meta_infos[sample][stream]
            meta_info = SampleMetaData(params={}, mask=mask)

            sdata = self._build_stream_data(sampling_idxs[sample], stream, mask)

            forcing_samples.samples[sample].add_stream_data(stream, sdata)
            forcing_samples.samples[sample].add_meta_info(stream, meta_info)

        forcing_samples.tokens_lens = get_tokens_lens(
            forcing_samples.streams, forcing_samples, self.forcing_window_len
        )

        return forcing_samples

    def _build_stream_data(self, sampling_idx: int, stream: str, input_mask: MaskData):
        """
        Build stream data equivalent to "network_input" mode for a particular stream.

        Neither target coordinates nor data is added to the StreamData instances,
        since it is not needed for processing in the encoder.
        """

        stream_data = StreamData(
            idx=sampling_idx,
            input_steps=self.forcing_window_len,
            output_steps=1,  # always only one output step
            healpix_cells=self.num_healpix_cells,
        )

        # adapted from _build_stream_data_input
        for step, idx in enumerate(  # TODO check correct semantics of step
            range(sampling_idx, sampling_idx - self.forcing_window_len, -1)
        ):
            # this stream's own lagged timeline, so the tokenizer stamps the window the data
            # is actually from rather than the window it is a forcing for (L5)
            time_win_source = self.handler(stream).window(idx)

            dataset_readers: list[DataReaderBase] = self.forcing_streams[stream]
            stream_info = dataset_readers[0].stream_info
            rdata = self._collect_forcing_data(dataset_readers, idx, time_win_source)
            # TODO filter channels
            token_data = self.tokenizer.get_tokens_windows(
                dataset_readers[0].stream_info, [rdata], True
            )[0]

            # TODO is this the intended behaviour => all(rdatas.is spoof)
            stream_data.source_is_spoof[step] = rdata.is_spoof

            # preprocess data for model input
            (source_cells, source_cells_lens) = self.tokenizer.get_source(
                stream_info,
                rdata,
                token_data,
                (time_win_source.start, time_win_source.end),
                input_mask,
            )

            stream_data.add_source(
                self.stage, step, rdata, source_cells_lens, source_cells, rdata.is_spoof
            )

        return stream_data

    def _collect_forcing_data(
        self, dataset_readers: list[DataReaderBase], idx: int, time_window: DTRange
    ) -> IOReaderData:
        """Collect forcing data for particular stream from all reader. Spoof if none."""
        rdatas = []
        for file_reader in dataset_readers:
            # shuffle = ds.stream_info.get("shuffle_source", False)
            # no shuffling for now (unclear how to achieve same shuffling as source data)
            shuffle = False
            rdata = (
                file_reader.get_source(idx)
                .shuffle(None, shuffle, -1)
                .remove_nan_coords_and_geoinfos()
            )
            rdata = dataclasses.replace(
                rdata,
                data=file_reader.normalize_source_channels(rdata.data),
                geoinfos=file_reader.normalize_geoinfos(rdata.geoinfos),
            )
            rdatas.append(rdata)

        combined_data = IOReaderData.combine(rdatas)
        if combined_data.is_empty():
            example_reader = dataset_readers[0]
            combined_data = spoof(
                self.healpix_lvl,
                time_window.start,
                example_reader.get_geoinfo_size(),
                len(example_reader.mean[example_reader.source_idx]),
            )

        return combined_data


class ForcingEngine(torch.nn.Module):
    def __init__(self, cf: Config, n_blocks=1):
        super().__init__()
        self.cf = cf

        blocks = []
        for _ in range(n_blocks):
            blocks.extend(self.get_block())

        self.blocks = torch.nn.ModuleList(blocks)

        # The blocks are residual and write into the pretrained latent at every rollout step,
        # so they have to start as a near-identity: with the default Linear init the freshly
        # built engine perturbs the latent enough to diverge a finetuning run once the LR
        # warmup peaks. Same treatment ForecastEngine gives its blocks in engines.py.
        def init_weights_final(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias, mean=0, std=0.001)

        for block in self.blocks:
            block.apply(init_weights_final)

    def get_block(self) -> list[torch.Module]:
        return [  # CrossAttention block, similiar to PerceiverIO
            MultiCrossAttentionHeadVarlen(
                # both X_q and X_kv share the same dimensionality & semantics
                dim_embed_q=self.cf.ae_global_dim_embed,
                dim_embed_kv=self.cf.ae_global_dim_embed,
                num_heads=self.cf.fe_num_heads,
                dim_head_proj=None,
                dropout_rate=self.cf.ae_adapter_dropout_rate,
                with_residual=True,
                with_qk_lnorm=True,  # TODO check what this is
                with_flash=self.cf.with_flash_attention,
                norm_type=self.cf.norm_type,  # how to norm x_q, x_kv and projected values
                norm_eps=self.cf.norm_eps,
                attention_dtype=get_dtype(self.cf.attention_dtype),
            ),
            MLP(
                dim_in=self.cf.ae_global_dim_embed,
                dim_out=self.cf.ae_global_dim_embed,
                with_residual=True,  # TODO is this needed? => already residual in Attention
                dropout_rate=self.cf.ae_adapter_dropout_rate,
                norm_type=self.cf.norm_type,  # how to norm input
                norm_eps=self.cf.mlp_norm_eps,
            ),
        ]

    def forward(self, latent_tokens, forcing_tokens):
        # MultiCrossAttentionHeadVarlen expects flattened varlen tokens + lens vectors.
        # Here we adapt from batched [B, T, D] tensors and restore shape afterwards.
        assert latent_tokens.ndim == 3, (
            f"Expected latent_tokens to be [B,T,D], got {latent_tokens.shape}"
        )
        assert forcing_tokens.ndim == 3, (
            f"Expected forcing_tokens to be [B,T,D], got {forcing_tokens.shape}"
        )

        batch_size, latent_len, dim_embed = latent_tokens.shape
        forcing_batch = forcing_tokens.shape[0]
        forcing_len = forcing_tokens.shape[1]

        if forcing_batch != batch_size:
            assert forcing_batch % batch_size == 0, (
                f"Incompatible batch sizes for forcing attention: latent B={batch_size}, "
                f"forcing B={forcing_batch}"
            )
            num_steps = forcing_batch // batch_size
            forcing_tokens = forcing_tokens.reshape(
                batch_size, num_steps, forcing_len, forcing_tokens.shape[-1]
            ).sum(dim=1)
            forcing_len = forcing_tokens.shape[1]

        latent_tokens_flat = latent_tokens.reshape(batch_size * latent_len, dim_embed)
        forcing_tokens_flat = forcing_tokens.reshape(
            batch_size * forcing_len, forcing_tokens.shape[-1]
        )

        latent_lens = torch.full(
            (batch_size + 1,), fill_value=latent_len, dtype=torch.int32, device=latent_tokens.device
        )
        forcing_lens = torch.full(
            (batch_size + 1,),
            fill_value=forcing_len,
            dtype=torch.int32,
            device=latent_tokens.device,
        )
        latent_lens[0] = 0
        forcing_lens[0] = 0

        for block in self.blocks:
            if isinstance(block, MultiCrossAttentionHeadVarlen):
                latent_tokens_flat = checkpoint(
                    block,
                    latent_tokens_flat,
                    forcing_tokens_flat,
                    latent_lens,
                    forcing_lens,
                    use_reentrant=False,
                )
            else:
                latent_tokens_flat = checkpoint(block, latent_tokens_flat, use_reentrant=False)

        return latent_tokens_flat.reshape(batch_size, latent_len, dim_embed)
