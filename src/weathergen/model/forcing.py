from __future__ import annotations
from typing import Any

import itertools as it

import torch
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.common.io import IOReaderData
from weathergen.datasets.batch import BatchSamples, SampleMetaData
from weathergen.datasets.data_reader_base import DataReaderBase, DTRange, TimeWindowHandler
from weathergen.datasets.masking import MaskData
from weathergen.datasets.stream_data import StreamData, spoof
from weathergen.datasets.tokenizer_masking import TokenizerMasking
from weathergen.datasets.utils import get_tokens_lens
from weathergen.model.attention import MultiCrossAttentionHeadVarlen
from weathergen.model.layers import MLP
from weathergen.model.model import Model, ModelOutput, ModelParams
from weathergen.model.utils import get_num_parameters
from weathergen.utils.utils import get_dtype


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
        source_samples: BatchSamples,
        dynamic_forcings: ForcingInput,
    ) -> ModelOutput:
        """Forward pass of the model

        Tokens are processed through the model components, which were defined in the create method.
        Args:
            model_params : Query and embedding parameters
            batch
        Returns:
            A list containing all prediction results
        """

        source_masks, source_sampling_idxs = self._get_source_masks_sample_idxs(source_samples)

        # output_idxs start with output_offset
        output_offset = source_samples.get_output_idxs()[0]

        output = ModelOutput(source_samples.get_output_len())

        tokens, posteriors = self.encoder(model_params, source_samples)
        output.add_latent_prediction(0, "posteriors", posteriors)

        # recover batch dimension and separate input_steps
        shape = (len(source_samples), source_samples.get_num_steps(), *tokens.shape[1:])
        # collapse along input step dimension
        tokens = tokens.reshape(shape).sum(axis=1)

        # Allow for pushforward trick TODO: enable
        p_fwd = self.cf.training_config.get("forecast", {}).get("pushforward", False)
        # roll-out in latent space, iterate and generate output over requested output steps
        for step in source_samples.get_output_idxs():
            forcing_idx = step - output_offset

            if self.forcing_engine and not dynamic_forcings.is_empty:
                # reembedd forcings
                forcing_sampling_idxs = [
                    sample_idx + forcing_idx for sample_idx in source_sampling_idxs
                ]
                forcings = dynamic_forcings.get_data(forcing_sampling_idxs, source_masks)
                forcings = forcings.to_device(tokens.device)
                forcing_tokens, _ = self.encoder(model_params, forcings)

                # combine forcings with current latent space
                tokens = self.forcing_engine(tokens, forcing_tokens)

            without_grad = p_fwd and self.training and step != max(source_samples.get_output_idxs())
            if without_grad:
                # Pushforward mode: advance tokens without grad; no decoding with torch.no_grad():
                tokens = self.forecast_engine(tokens, step, model_params.rope_coords)
                continue

            tokens = self.forecast_engine(tokens, forcing_idx, coords=model_params.rope_coords)
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
    def __init__(
        self,
        time_window_handler: TimeWindowHandler,
        forcing_streams: dict[str, list[DataReaderBase]],
        tokenizer: TokenizerMasking,
    ):
        self.forcing_window_len = 1
        self.forcing_streams = forcing_streams
        self.healpix_lvl = 5  # TODO infer from MSDS
        self.num_healpix_cells = 12 * 4**self.healpix_lvl

        self.time_window_handler = time_window_handler
        self.tokenizer = tokenizer
        self.tokenize_spacetime = True  # TODO hardcoded, do properly

    def is_empty(self) -> bool:
        return len(self.forcing_streams) == 0

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

        forcing_stream_infos = [readers[0].stream_info for readers in self.forcing_streams.values()]
        forcing_samples = BatchSamples(
            streams=forcing_stream_infos,
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
            forcing_stream_infos, forcing_samples, self.forcing_window_len
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
            time_win_source = self.time_window_handler.window(idx)

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

            stream_data.add_source(step, rdata, source_cells_lens, source_cells)

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
            rdata.data = file_reader.normalize_source_channels(rdata.data)
            rdata.geoinfos = file_reader.normalize_geoinfos(rdata.geoinfos)
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
    name: "ForcingEngine"  # TODO what the fuck is this, how is it used?

    def __init__(self, cf: Config, n_blocks=1):
        super().__init__()
        self.cf = cf
        
        blocks = []
        for _ in range(n_blocks):
            blocks.extend(self.get_block())

        self.blocks = torch.nn.ModuleList(blocks)

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
