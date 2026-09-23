# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Predictions are written with their own geometry, which may differ in rows from the target's.

A spoofed step (target window outside the dataset) or an inference_only run writes N prediction
rows and 0 target rows. These tests pin that at three levels: the chunk grouping in
`validation_io` (`_merge_steps` / `_group`), `OutputBatchData` extraction, and `write_output`.

Every row carries a tag that encodes (sample, step, row), in the data and in the coords, so a
row that lands in the wrong sample, or data that separates from its coords, is visible.
"""

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

import weathergen.utils.validation_io as vio
from weathergen.common.io import ItemKey, OutputBatchData, TimeRange
from weathergen.model.chunking import ChunkInfo
from weathergen.model.model import ModelOutput
from weathergen.train.target_and_aux_module_base import TargetAuxOutput

N_CH = 2
STREAM = "ocean"
T0 = np.datetime64("2023-01-01T00:00", "ns")


def tag(sample: int, step: int, row: int) -> float:
    return 1000.0 * sample + 100.0 * step + row


def times_of(tags) -> NDArray:
    """Valid times that encode the row tags, so times can be checked against data."""
    return T0 + np.asarray(tags).astype(np.int64).astype("timedelta64[m]")


def rows(lens: list[int], step: int) -> list[float]:
    """Tags of one step's rows, sample-major as the writer concatenates them."""
    return [tag(s, step, r) for s, n in enumerate(lens) for r in range(n)]


def step_arrays(lens: list[int], step: int):
    """One stream's per-step arrays: preds (ens, n, ch), targets (n, ch), coords, times."""
    t = np.asarray(rows(lens, step), dtype=np.float32)
    data = np.repeat(t[:, None], N_CH, axis=1)
    coords = np.stack([t, -t], axis=1)
    times = times_of(t)
    return data[None], data, coords, times


def expected_group(lens_per_step: list[list[int]], steps: list[int]) -> list[float]:
    """Sample-major merge: every step of sample 0, then every step of sample 1."""
    n_samples = len(lens_per_step[0])
    return [
        tag(s, step, r)
        for s in range(n_samples)
        for lens, step in zip(lens_per_step, steps, strict=True)
        for r in range(lens[s])
    ]


# 2 samples x 3 steps. Predictions always cover the full geometry; the target is full at step 0,
# absent at step 1 (spoofed / target-less) and differently sized per sample at step 2.
PREDS_LENS = [[3, 2], [3, 2], [3, 2]]
TARGETS_LENS = [[3, 2], [0, 0], [1, 4]]


# ------------------------------------------------------------------------ _merge_steps / _group


def test_merge_steps_is_sample_major_with_empty_step():
    parts = [np.asarray(rows(lens, i)) for i, lens in enumerate(TARGETS_LENS)]
    merged = vio._merge_steps(parts, TARGETS_LENS, axis=0)
    assert merged.tolist() == expected_group(TARGETS_LENS, [0, 1, 2])


def test_merge_steps_along_ensemble_axis():
    parts = [step_arrays(lens, i)[0] for i, lens in enumerate(PREDS_LENS)]
    merged = vio._merge_steps(parts, PREDS_LENS, axis=1)
    assert merged.shape == (1, 15, N_CH)
    assert merged[0, :, 0].tolist() == expected_group(PREDS_LENS, [0, 1, 2])


@pytest.mark.parametrize(
    ("forecast_offset", "chunk_size", "expected"),
    [
        (0, 3, {0: [0, 1, 2]}),
        (1, 2, {0: [0], 1: [1, 2]}),
        (1, 3, {0: [0], 1: [1, 2]}),
    ],
)
def test_chunk_groups(forecast_offset, chunk_size, expected):
    assert vio._chunk_groups([0, 1, 2], forecast_offset, chunk_size) == expected


@pytest.mark.parametrize(("forecast_offset", "chunk_size"), [(0, 3), (1, 2)])
def test_group_slices_preds_and_targets_by_their_own_lens(forecast_offset, chunk_size):
    steps = [0, 1, 2]
    per_step = [
        step_arrays(p, i) + step_arrays(t, i)
        for i, (p, t) in enumerate(zip(PREDS_LENS, TARGETS_LENS, strict=True))
    ]
    # [step][stream] as write_output holds them, with a single stream
    preds = [[a[0]] for a in per_step]
    preds_coords = [[a[2]] for a in per_step]
    preds_times = [[a[3]] for a in per_step]
    targets = [[a[5]] for a in per_step]
    targets_coords = [[a[6]] for a in per_step]
    targets_times = [[a[7]] for a in per_step]
    preds_lens = [[lens] for lens in PREDS_LENS]
    targets_lens = [[lens] for lens in TARGETS_LENS]

    groups = vio._chunk_groups(steps, forecast_offset, chunk_size)
    g_preds = vio._group(preds, preds_lens, 1, groups)
    g_preds_coords = vio._group(preds_coords, preds_lens, 0, groups)
    g_preds_times = vio._group(preds_times, preds_lens, 0, groups)
    g_targets = vio._group(targets, targets_lens, 0, groups)
    g_targets_coords = vio._group(targets_coords, targets_lens, 0, groups)
    g_targets_times = vio._group(targets_times, targets_lens, 0, groups)
    g_preds_lens = vio._group_lens(preds_lens, groups)
    g_targets_lens = vio._group_lens(targets_lens, groups)

    assert len(g_preds) == len(groups)
    for g, positions in enumerate(groups.values()):
        p_lens = [PREDS_LENS[p] for p in positions]
        t_lens = [TARGETS_LENS[p] for p in positions]
        exp_p = expected_group(p_lens, positions)
        exp_t = expected_group(t_lens, positions)

        assert g_preds_lens[g][0] == [sum(ls) for ls in zip(*p_lens, strict=True)]
        assert g_targets_lens[g][0] == [sum(ls) for ls in zip(*t_lens, strict=True)]

        assert g_preds[g][0][0, :, 0].tolist() == exp_p
        assert g_preds_coords[g][0][:, 0].tolist() == exp_p
        assert g_preds_times[g][0].tolist() == times_of(exp_p).tolist()
        assert g_targets[g][0][:, 0].tolist() == exp_t
        assert g_targets_coords[g][0][:, 0].tolist() == exp_t
        assert g_targets_times[g][0].tolist() == times_of(exp_t).tolist()

        # each sample's rows are one contiguous run, located by the merged lens
        for lens, merged in ((g_preds_lens, g_preds_coords), (g_targets_lens, g_targets_coords)):
            off = np.cumsum([0, *lens[g][0]])
            for s in range(2):
                run = merged[g][0][off[s] : off[s + 1], 0]
                assert ((run // 1000) == s).all()


# ------------------------------------------------------------------------------ OutputBatchData


def make_batch_data(preds_lens, targets_lens, separate_preds_geometry=True, forecast_offset=1):
    """OutputBatchData with one stream, laid out as write_output hands it over."""
    per_step = [
        step_arrays(p, i) + step_arrays(t, i)
        for i, (p, t) in enumerate(zip(preds_lens, targets_lens, strict=True))
    ]
    n_samples = len(preds_lens[0])
    source = SimpleNamespace(
        data=np.zeros((1, N_CH), dtype=np.float32),
        datetimes=np.array([T0]),
        coords=np.zeros((1, 2), dtype=np.float32),
        geoinfos=np.zeros((1, 0), dtype=np.float32),
    )
    kwargs = {}
    if separate_preds_geometry:
        kwargs = {
            "preds_coords": [[a[2]] for a in per_step],
            "preds_times": [[a[3]] for a in per_step],
            "preds_lens": [[lens] for lens in preds_lens],
        }
    return OutputBatchData(
        sources=[[source] for _ in range(n_samples)],
        source_intervals=[TimeRange(T0, T0 + np.timedelta64(6, "h"))] * n_samples,
        targets=[[a[5]] for a in per_step],
        predictions=[[a[0]] for a in per_step],
        targets_coords=[[a[6]] for a in per_step],
        targets_times=[[a[7]] for a in per_step],
        targets_lens=[[lens] for lens in targets_lens],
        streams={STREAM: 0},
        target_channels=[[f"c{i}" for i in range(N_CH)]],
        source_channels=[[f"c{i}" for i in range(N_CH)]],
        geoinfo_channels=[[]],
        sample_start=0,
        forecast_offset=forecast_offset,
        forecast_steps=list(range(len(preds_lens))),
        **kwargs,
    )


def test_output_batch_data_separate_prediction_geometry():
    # step 0 is source only, step 1 has a target, step 2 is spoofed: predictions only
    preds_lens = [[0, 0], [3, 2], [3, 2]]
    targets_lens = [[0, 0], [3, 2], [0, 0]]
    data = make_batch_data(preds_lens, targets_lens)

    items = {(it.key.sample, it.key.forecast_step): it for it in data.items()}
    assert set(items) == {(s, f) for s in range(2) for f in range(3)}

    for s in range(2):
        assert [d.name for d in items[(s, 0)].datasets] == ["source"]

        normal = items[(s, 1)]
        exp = [tag(s, 1, r) for r in range(preds_lens[1][s])]
        assert normal.target.data[:, 0].tolist() == exp
        assert normal.target.coords[:, 0].tolist() == exp
        assert normal.prediction.data.shape == (len(exp), N_CH, 1)
        assert normal.prediction.data[:, 0, 0].tolist() == exp
        np.testing.assert_array_equal(normal.prediction.coords, normal.target.coords)
        np.testing.assert_array_equal(normal.prediction.times, normal.target.times)

        spoof = items[(s, 2)]
        assert spoof.target.data.shape == (0, N_CH)
        assert spoof.target.coords.shape == (0, 2)
        assert len(spoof.target.times) == 0
        exp = [tag(s, 2, r) for r in range(preds_lens[2][s])]
        assert spoof.prediction.data.shape == (len(exp), N_CH, 1)
        assert spoof.prediction.data[:, 0, 0].tolist() == exp
        assert spoof.prediction.coords[:, 0].tolist() == exp
        assert spoof.prediction.coords[:, 1].tolist() == [-e for e in exp]
        assert spoof.prediction.times.tolist() == times_of(exp).tolist()


def test_output_batch_data_uneven_sample_lens():
    # different per-sample splits for predictions and targets within one step
    data = make_batch_data([[0, 0], [1, 4]], [[0, 0], [3, 0]])
    for s, (n_p, n_t) in enumerate([(1, 3), (4, 0)]):
        item = data.extract(ItemKey(s, 1, STREAM))
        assert item.prediction.coords[:, 0].tolist() == [tag(s, 1, r) for r in range(n_p)]
        assert item.target.coords[:, 0].tolist() == [tag(s, 1, r) for r in range(n_t)]


def test_output_batch_data_falls_back_to_target_geometry():
    lens = [[0, 0], [3, 2], [1, 4]]
    data = make_batch_data(lens, lens, separate_preds_geometry=False)
    assert data.preds_coords is None and data.preds_lens is None
    for s in range(2):
        for f in (1, 2):
            item = data.extract(ItemKey(s, f, STREAM))
            exp = [tag(s, f, r) for r in range(lens[f][s])]
            assert item.prediction.data[:, 0, 0].tolist() == exp
            assert item.target.data[:, 0].tolist() == exp
            np.testing.assert_array_equal(item.prediction.coords, item.target.coords)
            np.testing.assert_array_equal(item.prediction.times, item.target.times)


# --------------------------------------------------------------------------------- write_output

# per-sample point counts; steps 1..3 carry output (forecast_offset 1)
N_POINTS = [3, 2]
OUTPUT_IDXS = [1, 2, 3]


class _Cfg(dict):
    __getattr__ = dict.__getitem__


def _point_tags(s: int, step: int) -> NDArray:
    return np.asarray([tag(s, step, r) for r in range(N_POINTS[s])], dtype=np.float32)


def _make_inputs(kind: str, spoof_step: int | None, spoof_samples=(0, 1)):
    """Batch, target aux and a filler for ModelOutput, for one stream and 2 samples.

    kind "target": normal targets, spoofed at `spoof_step` for the samples in `spoof_samples`
    (whose target then holds filler that must not be written); "inference_only": no target half.
    Predictions and targets carry the point tag in channel 0 in *storage* order; they and the
    coords arrive permuted by the inverse of idxs_inv, so that a writer which forgot to apply
    idxs_inv to any of them would split data from coords.
    """
    samples = []
    for s in range(2):
        sd = SimpleNamespace(
            source_raw=[
                SimpleNamespace(
                    data=np.zeros((1, N_CH), dtype=np.float32),
                    datetimes=np.array([T0]),
                    coords=np.zeros((1, 2), dtype=np.float32),
                    geoinfos=np.zeros((1, 0), dtype=np.float32),
                )
            ],
            sample_idx=s,
        )
        samples.append(SimpleNamespace(streams_data={STREAM: sd}))
    batch = SimpleNamespace(
        get_output_idxs=lambda: OUTPUT_IDXS,
        get_source_samples=lambda: SimpleNamespace(get_samples=lambda: samples),
    )

    aux = TargetAuxOutput(OUTPUT_IDXS[-1] + 1, OUTPUT_IDXS)
    preds = {}
    for step in OUTPUT_IDXS:
        entry = {k: [] for k in ("target", "target_coords", "target_times", "idxs_inv")}
        entry["is_spoof"] = [step == spoof_step and s in spoof_samples for s in range(2)]
        preds[step] = []
        for s in range(2):
            t = _point_tags(s, step)
            n = len(t)
            # token order is reversed storage order; idxs_inv maps it back
            perm = torch.arange(n - 1, -1, -1)
            tok = torch.from_numpy(t)[perm]
            vals = tok[:, None].repeat(1, N_CH)
            preds[step].append(vals[None].clone())
            entry["target_coords"].append(torch.stack([tok, -tok], dim=1))
            entry["target_times"].append(times_of(t)[perm.numpy()])
            if kind == "inference_only":
                # no target half: empty target, empty idxs_inv, rows stay in token order
                entry["target"].append(torch.zeros(0))
                entry["idxs_inv"].append(torch.zeros(0, dtype=torch.int64))
            else:
                spoofed = entry["is_spoof"][s]
                entry["target"].append(torch.full_like(vals, -1.0) if spoofed else vals.clone())
                entry["idxs_inv"].append(perm)  # the reversal is its own inverse
        aux.add_physical_target(step, STREAM, entry)
    return batch, {"mse": aux}, preds


@pytest.fixture
def captured(monkeypatch):
    """Capture every OutputBatchData subset write_output would write."""
    written = []

    class _Writer:
        def write_zarr(self, item):
            written.append(item)

    @contextlib.contextmanager
    def fake_writer(path):
        yield _Writer()

    monkeypatch.setattr(vio, "zarrio_writer", fake_writer)
    monkeypatch.setattr(vio.config, "get_path_results", lambda cf, mini_epoch: None)
    return written


def _run_write_output(kind, spoof_step, spoof_samples, chunk_size, captured):
    cf = _Cfg(
        streams={
            STREAM: _Cfg(
                val_target_channels=[f"c{i}" for i in range(N_CH)],
                val_source_channels=[f"c{i}" for i in range(N_CH)],
            )
        }
    )
    val_cfg = _Cfg(
        losses={"mse": _Cfg(type="LossPhysical")},
        forecast={"chunk_size": chunk_size},
        output={"streams": None},
        start_date=T0,
        end_date=T0 + np.timedelta64(30, "D"),
        time_window_len=np.timedelta64(6, "h"),
        time_window_step=np.timedelta64(6, "h"),
    )
    batch, aux, preds = _make_inputs(kind, spoof_step, spoof_samples)

    per_store_step = {}
    for tile in ChunkInfo.tiles(OUTPUT_IDXS, chunk_size):
        out = ModelOutput(tile, batch.get_source_samples())
        for step in tile.steps:
            out.add_physical_prediction(out.chunk_idx(step), STREAM, preds[step])
        captured.clear()
        vio.write_output(cf, val_cfg, 2, 0, 0, lambda _s, x: x, batch, out, aux)
        for item in captured:
            per_store_step[(item.key.sample, item.key.forecast_step)] = item
    return per_store_step


def _store_steps(chunk_size):
    """Global forecast steps making up each store step (0 is the source-only step)."""
    groups = {0: [0]}
    for step in OUTPUT_IDXS:
        groups.setdefault(1 + (step - 1) // chunk_size, []).append(step)
    return groups


@pytest.mark.parametrize("chunk_size", [1, 2])
@pytest.mark.parametrize(
    ("kind", "spoof_step", "spoof_samples"),
    [
        ("target", None, ()),
        ("target", 2, (0, 1)),
        # samples of one batch have different init times, so only some may run off the data
        ("target", 2, (1,)),
        ("target", 2, (0,)),
        ("inference_only", None, ()),
    ],
    ids=["normal", "spoofed", "spoofed_sample1", "spoofed_sample0", "inference_only"],
)
def test_write_output_prediction_geometry(kind, spoof_step, spoof_samples, chunk_size, captured):
    items = _run_write_output(kind, spoof_step, spoof_samples, chunk_size, captured)
    store_steps = _store_steps(chunk_size)
    assert set(items) == {(s, f) for s in range(2) for f in store_steps}

    for s in range(2):
        assert [d.name for d in items[(s, 0)].datasets] == ["source"]
        for f, steps in store_steps.items():
            if f == 0:
                continue
            item = items[(s, f)]
            # without a target half nothing reorders, so rows stay in (reversed) token order
            order = -1 if kind == "inference_only" else 1
            exp_p = [t for step in steps for t in _point_tags(s, step)[::order].tolist()]
            has_target = [
                kind == "target" and not (step == spoof_step and s in spoof_samples)
                for step in steps
            ]
            exp_t = [
                t
                for step, ht in zip(steps, has_target, strict=True)
                if ht
                for t in _point_tags(s, step).tolist()
            ]

            # predictions: every step in full, data aligned with its own coords and times
            assert item.prediction.data.shape == (len(exp_p), N_CH, 1)
            assert item.prediction.data[:, 0, 0].tolist() == exp_p
            assert item.prediction.coords[:, 0].tolist() == exp_p
            assert item.prediction.coords[:, 1].tolist() == [-t for t in exp_p]
            assert item.prediction.times.tolist() == times_of(exp_p).tolist()

            # targets: only the steps that have one, same alignment
            assert item.target.data.shape == (len(exp_t), N_CH)
            assert item.target.data[:, 0].tolist() == exp_t
            assert item.target.coords[:, 0].tolist() == exp_t
            assert item.target.times.tolist() == times_of(exp_t).tolist()
