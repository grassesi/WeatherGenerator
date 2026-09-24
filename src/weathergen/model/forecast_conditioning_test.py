# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CPU tests of the forecast conditioning pieces that need neither a GPU nor data on disk."""

import dataclasses

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.common.io import IOReaderData
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.datasets.elevation import ElevatingReader
from weathergen.datasets.elevation_test import (
    END,
    FORECAST_STEP,
    START,
    WINDOW_LEN,
    WINDOW_STEP,
    FakeReader,
    make_stream_info,
)
from weathergen.datasets.utils import healpix_verts_rots, r3tos2
from weathergen.model.forcing import ForcingEngine, ForcingInput
from weathergen.model.forecast_conditioning import (
    AdaLNConditioning,
    build_forecast_conditioning,
    check_conditioning_keys,
    forcing_cell_index,
    scatter_to_cells,
)
from weathergen.model.layers import MLP
from weathergen.model.norms import ModulatedLayerNorm, ZeroLinear
from weathergen.train.utils import TRAIN

####################################################################################################
# pooling


def test_scatter_to_cells_ignores_nan_per_channel():
    # cell 0: two points, the second has a NaN in channel 1 only; cell 1: all NaN; cell 2: empty
    values = torch.tensor([[1.0, 10.0], [3.0, float("nan")], [float("nan"), float("nan")]])
    cell_idx = torch.tensor([0, 0, 1])

    cell_values, cell_valid = scatter_to_cells(values, cell_idx, num_cells=3)

    torch.testing.assert_close(cell_values, torch.tensor([[2.0, 10.0], [0.0, 0.0], [0.0, 0.0]]))
    torch.testing.assert_close(cell_valid, torch.tensor([[1.0, 0.5], [0.0, 0.0], [0.0, 0.0]]))


@pytest.mark.parametrize("healpix_level", [3, 5])
def test_cell_index_maps_every_cell_centre_to_its_own_cell(healpix_level):
    """Cell i of the pooled field must be latent cell i, i.e. the nested cell the tokenizer uses.

    The centres come in healpy's longitude convention [0, 360); the tokenizer's is lon + 180, so
    a centre at healpy longitude L is a point at lon = L - 180 in the data's coordinates.
    """
    verts, _ = healpix_verts_rots(healpix_level, 0.5, 0.5)
    lat_lon = torch.rad2deg(r3tos2(verts)).numpy()
    lat, lon = lat_lon[:, 0], lat_lon[:, 1]
    lon = np.where(lon < 0, lon + 360.0, lon) - 180.0

    idx = forcing_cell_index(np.stack([lat, lon], axis=1), healpix_level)

    np.testing.assert_array_equal(idx, np.arange(12 * 4**healpix_level))


def _conditioning(streams=("A", "B"), channels=(1, 2), num_queries=1, dim_cond=8, level=1):
    return AdaLNConditioning(
        list(streams),
        list(channels),
        healpix_level=level,
        num_queries=num_queries,
        dim_cond=dim_cond,
    )


def _field(coords, data, is_spoof=False):
    n = len(coords)
    return IOReaderData(
        coords=np.asarray(coords, dtype=np.float32),
        geoinfos=np.zeros((n, 0), dtype=np.float32),
        data=np.asarray(data, dtype=np.float32),
        datetimes=np.full((n,), START, dtype="datetime64[ns]"),
        is_spoof=is_spoof,
    )


def test_pool_fields_concatenates_streams_and_drops_spoofs():
    cond = _conditioning()
    # well inside a cell: (0, 0) is a cell corner, where float32 and float64 disagree
    coords = [[10.3, 20.7], [10.3, 20.7]]
    cell = int(forcing_cell_index(np.asarray(coords, dtype=np.float32), cond.healpix_level)[0])
    fields = [
        {"A": _field(coords, [[1.0], [3.0]]), "B": _field(coords, [[5.0, 6.0], [7.0, 8.0]])},
        {"A": _field(coords, [[1.0], [3.0]], is_spoof=True), "B": _field(coords, [[1, 1], [1, 1]])},
    ]

    pooled = cond.pool_fields(fields, torch.device("cpu"))

    # [values A | values B | valid A | valid B]
    assert pooled.shape == (2, cond.num_cells, 2 * 3)
    torch.testing.assert_close(pooled[0, cell], torch.tensor([2.0, 6.0, 7.0, 1.0, 1.0, 1.0]))
    # a spoofed stream is absent: no value, no validity, while the other stream is still there
    torch.testing.assert_close(pooled[1, cell], torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 1.0]))
    # cells without points are absent
    assert pooled[0, [c for c in range(cond.num_cells) if c != cell]].abs().sum() == 0


def test_forward_expands_to_the_token_layout_with_unconditioned_aux_tokens():
    cond = _conditioning(num_queries=2)
    num_aux = 3
    tokens = torch.randn(2, num_aux + cond.num_cells * 2, 4)
    encoded = torch.randn(2, cond.num_cells, 8)

    out_tokens, aux = cond(tokens, encoded)

    assert out_tokens is tokens
    assert aux.shape == (2, tokens.shape[1], 8)
    assert aux[:, :num_aux].abs().sum() == 0
    # cell c owns tokens num_aux + 2c and num_aux + 2c + 1
    torch.testing.assert_close(aux[:, num_aux::2], encoded)
    torch.testing.assert_close(aux[:, num_aux + 1 :: 2], encoded)


####################################################################################################
# modulation


def _reset_like_model(module):
    """What `Model.reset_parameters` does after an FSDP `to_empty()`."""
    for m in module.modules():
        if isinstance(m, torch.nn.Linear | torch.nn.LayerNorm):
            m.reset_parameters()


def test_zero_linear_stays_zero_on_every_reset():
    norm = ModulatedLayerNorm(16, 4)
    _reset_like_model(norm)
    assert torch.count_nonzero(norm.modulation.weight) == 0

    torch.nn.init.normal_(norm.modulation.weight)
    norm.modulation.reset_parameters()
    assert torch.count_nonzero(norm.modulation.weight) == 0
    assert norm.modulation.bias is None


def test_forecast_engine_init_leaves_the_modulation_at_zero():
    """The FE's 0.001 init touches every Linear; a ZeroLinear must be skipped."""
    from weathergen.model.engines import ForecastingEngine

    cf = OmegaConf.create(
        {
            "fe_num_blocks": 1,
            "forecast_att_dense_rate": 1.0,
            "ae_global_dim_embed": 16,
            "fe_num_heads": 2,
            "fe_dropout_rate": 0.0,
            "fe_with_qk_lnorm": True,
            "with_flash_attention": True,
            "norm_type": "LayerNorm",
            "norm_eps": 1e-5,
            "mlp_norm_eps": 1e-5,
            "attention_dtype": "fp32",
            "ae_local_num_queries": 1,
            "ae_global_block_factor": 1,
        }
    )
    mode_cfg = OmegaConf.create({"forecast": {"policy": "fixed"}})
    try:
        fe = ForecastingEngine(cf, mode_cfg, num_healpix_cells=12, dim_modulation=4)
    except (AssertionError, RuntimeError, ImportError) as e:  # flash attention on a CPU node
        pytest.skip(f"ForecastingEngine not constructible here: {e}")

    modulations = [m for m in fe.modules() if isinstance(m, ZeroLinear)]
    assert len(modulations) == 2  # one attention block, one MLP
    assert all(torch.count_nonzero(m.weight) == 0 for m in modulations)


def test_zero_modulation_is_the_unmodulated_block():
    torch.manual_seed(0)
    plain = MLP(16, 16, with_residual=True)
    modulated = MLP(16, 16, with_residual=True, dim_modulation=4)
    missing, unexpected = modulated.load_state_dict(plain.state_dict(), strict=False)
    assert missing == ["layers.0.modulation.weight"] and unexpected == []

    x = torch.randn(2, 5, 16)
    aux = torch.randn(2, 5, 4)
    assert torch.equal(modulated(x, None, aux), plain(x, None, None))


def test_modulation_leaves_tokens_with_zero_conditioning_alone():
    torch.manual_seed(0)
    block = MLP(16, 16, with_residual=True, dim_modulation=4)
    torch.nn.init.normal_(block.layers[0].modulation.weight)
    x = torch.randn(1, 5, 16)
    aux = torch.randn(1, 5, 4)
    aux[:, :2] = 0.0  # e.g. register/class tokens

    out, ref = block(x, None, aux), block(x, None, None)

    assert torch.equal(out[:, :2], ref[:, :2])
    assert not torch.allclose(out[:, 2:], ref[:, 2:])


####################################################################################################
# selection and the no-mixing guard


def _cf(streams, conditioning=None, ffe_num_blocks=0):
    conf = {"streams": streams, "ffe_num_blocks": ffe_num_blocks, "healpix_level": 1}
    conf["ae_local_num_queries"] = 1
    if conditioning is not None:
        conf["forecast_conditioning"] = conditioning
    return OmegaConf.create(conf)


FORCED_STREAM = {"S": {"is_dynamic_forcing": True, "train_source_channels": ["sst"]}}


def test_a_config_without_the_block_is_the_cross_attention_engine():
    engine = build_forecast_conditioning(_cf(FORCED_STREAM))
    assert isinstance(engine, ForcingEngine)
    assert engine.fe_modulation_dim is None


def test_adaln_is_selected_and_sized_from_the_forcing_channels():
    cf = _cf(FORCED_STREAM, {"type": "adaln", "dim_cond": 8})
    cond = build_forecast_conditioning(cf, {"S": 3, "OTHER": 5})
    assert isinstance(cond, AdaLNConditioning)
    assert cond.stream_channels == [3] and cond.fe_modulation_dim == 8


def test_cross_attention_refuses_a_forcing_stream_that_is_not_an_encoder_input():
    streams = {
        "S": {"is_dynamic_forcing": True, "diagnostic": True, "train_source_channels": ["sst"]}
    }
    with pytest.raises(ValueError, match="not an encoder input"):
        build_forecast_conditioning(_cf(streams))
    # AdaLN never encodes the forcing, so the same stream is fine there
    build_forecast_conditioning(_cf(streams, {"type": "adaln"}), {"S": 1})


XATTN_KEYS = ["forcing_engine.blocks.0.proj_heads_q.weight", "encoder.x.weight"]
ADALN_KEYS = [
    "forcing_engine.embed.mlp.0.weight",
    "forecast_engine.fe_blocks.0.lnorm.modulation.weight",
    "encoder.x.weight",
]


def test_checkpoints_of_the_two_methods_are_not_interchangeable():
    check_conditioning_keys("cross_attention", XATTN_KEYS)
    check_conditioning_keys("adaln", ADALN_KEYS)
    with pytest.raises(ValueError, match="trained with forecast conditioning 'cross_attention'"):
        check_conditioning_keys("adaln", XATTN_KEYS)
    with pytest.raises(ValueError, match="trained with forecast conditioning 'adaln'"):
        check_conditioning_keys("cross_attention", ADALN_KEYS)
    # the DiT decoder blocks' own zero-init modulation is not AdaLN conditioning
    check_conditioning_keys("cross_attention", ["decoder.0.adaLN_modulation.1.weight"])


####################################################################################################
# reading through ForcingInput


def test_get_fields_reads_through_the_stream_wrappers():
    """What AdaLN is fed goes through the same reader stack as the tokenized forcing."""
    handler = TimeWindowHandler(START, END, WINDOW_LEN, WINDOW_STEP)
    info = make_stream_info({"sst": {"offset": 2.0, "step": 0}})
    reader = ElevatingReader(FakeReader(info), START, FORECAST_STEP)
    forcings = ForcingInput(TRAIN, handler, {"FAKE": [reader]}, tokenizer=None)

    fields = forcings.get_fields([7, 8])

    assert len(fields) == 2 and set(fields[0]) == {"FAKE"}
    data = fields[0]["FAKE"].data
    # FakeReader's value is 100 * channel index; sst (channel 0) is elevated by 2
    np.testing.assert_array_equal(data[0], [2.0, 100.0, 200.0])
    assert not fields[0]["FAKE"].is_spoof


def test_get_fields_spoofs_an_empty_window():
    handler = TimeWindowHandler(START, END, WINDOW_LEN, WINDOW_STEP)
    reader = FakeReader(make_stream_info(), num_points=0)
    forcings = ForcingInput(TRAIN, handler, {"FAKE": [reader]}, tokenizer=None)

    (sample,) = forcings.get_fields([3])

    assert sample["FAKE"].is_spoof
    assert dataclasses.asdict(sample["FAKE"])["data"].shape[1] == 3


def test_the_trainer_reads_each_stage_forcing_on_its_own_dataset():
    """Validation used to read its forcing through the training dataset's timeline."""
    import types

    from weathergen.train.trainer import Trainer
    from weathergen.train.utils import VAL

    def dataset(start):
        handler = TimeWindowHandler(start, END, WINDOW_LEN, WINDOW_STEP)
        reader = FakeReader(make_stream_info())
        streams = {
            "FAKE": types.SimpleNamespace(readers=[reader], info={"is_dynamic_forcing": True}),
            "OTHER": types.SimpleNamespace(readers=[reader], info={}),
        }
        return types.SimpleNamespace(
            time_window_handler=handler, streams_datasets=streams, tokenizer=None
        )

    val_start = START + np.timedelta64(10, "D")
    mode_cfg = OmegaConf.create({"forecast": {"offset": 1}})
    forcings = Trainer._forcing_input(dataset(val_start), VAL, mode_cfg)

    assert forcings.stage == VAL
    assert list(forcings.forcing_streams) == ["FAKE"]
    assert forcings.time_window_handler.t_start == val_start
