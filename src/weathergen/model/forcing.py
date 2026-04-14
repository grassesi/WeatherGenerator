from __future__ import annotations

import itertools as it

import torch
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
#from weathergen.common.data import TimeWindowHandler
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
from weathergen.utils.utils import get_dtype


class ForcedModel(Model):
    def __init__(self, cf: Config, sources_size, targets_num_channels, targets_coords_size):
        super().__init__(cf, sources_size, targets_num_channels, targets_coords_size)

        self.forcing_engine = ForcingEngine(self.cf, self.num_healpix_cells)

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

        output = ModelOutput(source_samples.get_output_len())
        source_masks = [
            {stream: meta_info.mask for stream, meta_info in sample.meta_info.items()}
            for sample in source_samples.samples
        ]

        _example_sample_data = source_samples.samples[0].streams_data
        _example_stream = list(_example_sample_data.keys())[0]
        source_sampling_idx = 0 #_example_sample_data[_example_stream].idx
        # output_idxs start with output_offset
        output_offset = source_samples.get_output_idxs()[0]

        tokens, posteriors = self.encoder(model_params, source_samples)
        output.add_latent_prediction(0, "posteriors", posteriors)

        # recover batch dimension and separate input_steps
        shape = (len(source_samples), source_samples.get_num_steps(), *tokens.shape[1:])
        # collapse along input step dimension
        tokens = tokens.reshape(shape).sum(axis=1)

        # roll-out in latent space, iterate and generate output over requested output steps
        for step in source_samples.get_output_idxs():
            if self.forcing_engine:
                # reembedd forcings
                forcing_idx = source_sampling_idx + step - output_offset
                forcings = dynamic_forcings.get_data(source_sampling_idx, source_masks)
                forcing_tokens = self.encoder(model_params, forcings)

                # combine forcings with current latent space
                tokens = self.forcing_engine(tokens, forcing_tokens)

            # apply forecasting engine (if present)
            if self.forecast_engine:
                tokens = self.forecast_engine(tokens, forcing_idx, coords=model_params.rope_coords)

            # decoder predictions
            output = self.predict_decoders(model_params, step, tokens, source_samples, output)
            # latent predictions (raw and with SSL heads)
            output = self.predict_latent(model_params, step, tokens, source_samples, output)

        return output


class ForcingInput:
    def __init__(
        self,
        time_window_handler: TimeWindowHandler,
        forcing_streams: dict[str, DataReaderBase],
        tokenizer: TokenizerMasking,
    ):
        self.forcing_window_len = 1
        self.forcing_streams = forcing_streams
        self.healpix_lvl = 5  # TODO infer from MSDS
        self.num_healpix_cells = 12 * 4**self.healpix_lvl

        self.time_widow_handler = time_window_handler
        self.tokenizer = tokenizer
        self.tokenize_spacetime = True  # TODO hardcoded, do properly

    def get_data(
        self, sampling_idx: int, meta_infos: list[dict[str, SampleMetaData]]
    ) -> BatchSamples:
        """
        Sample all data sources for the input window corresponding to a given output step.

        Args:
          step: Output step to retrieve forcing sources for.

        Returns: Data that can be ingested by the Encoder.
        """

        samples = range(len(meta_infos))
        forcing_samples = BatchSamples(
            streams=[readers[0].stream_info for readers in self.forcing_streams],
            num_samples=len(samples),
            output_steps=1,
            output_idxs=None,  # not needed, since not used in encoder
        )

        for stream, sample in it.product(self.forcing_streams, samples):
            meta_info = meta_infos[sample][stream]
            sdata = self._build_stream_data(sampling_idx, stream, meta_info.mask)

            forcing_samples.samples[sample].add_stream_data(stream.name, sdata)
            forcing_samples.samples[sample].add_meta_info(stream.name, meta_info)

        print("self.forcing_streams:", self.forcing_streams)
        print("forcing_samples:", forcing_samples)
        print("self.forcing_window_len:", self.forcing_window_len)

        forcing_samples.tokens_lens = get_tokens_lens(
            self.forcing_streams, forcing_samples, self.forcing_window_len
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
            stream_data.source_is_spoof = rdata.is_spoof

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

        block = [  # CrossAttention block, similiar to PerceiverIO
            MultiCrossAttentionHeadVarlen(
                # both X_q and X_kv share the same dimensionality & semantics
                dim_embed_q=self.cf.ae_global_dim_embed,
                dim_embed_kv=self.cf.ae_global_dim_embed,
                num_heads=self.cf.fe_num_heads,
                dim_head_proj=self.cf.ae_global_dim_embed,  # TODO check what this is
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

        self.blocks = torch.nn.ModuleList(block * n_blocks)

    def forward(self, latent_tokens, forcing_tokens):
        for block in self.blocks:
            latent_tokens = checkpoint(block, latent_tokens, forcing_tokens)

        return latent_tokens
