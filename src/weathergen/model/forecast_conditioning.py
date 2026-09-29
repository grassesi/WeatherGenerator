# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
How a dynamic forcing conditions the forecast of `ForcedModel`.

`ForecastConditioning` is the interface `ForcedModel` drives once per rollout step: `encode`
reads and embeds the forcing for that step, `forward` applies it to the latent tokens and/or
returns a per-token conditioning the forecast engine is modulated by. Every implementation reads
through the component's `ForcingInput`, so the per-stream lag, a coupled producer's predictions
(`DataReaderCoupling`) and an `ElevatingReader` reach all of them the same way.

The implementation is chosen by `forecast_conditioning.type` in the config; a config without
the block is the reembed + cross-attention `ForcingEngine` (`forcing.py`) sized by
`ffe_num_blocks`, i.e. exactly the model every existing checkpoint was trained as. The methods
differ in how the forcing enters and are not interchangeable: `load_model` refuses a checkpoint
carrying another method's forcing weights.

Both methods take the same optional `forecast_conditioning.time` block: the predicted window's
time (`TimeConditioning`) is broadcast to every latent token and modulates every
forecast-engine block's pre-norm. The time path is identical in both, so its weights do not
identify a method.

`AdaLNConditioning` is a port of the `adaln_local` mode of `ForcingInjection`
(ankitpatnala/WeatherGenerator, branch develop-hclimrep_forcings): the forcing point cloud is
pooled onto the latent HEALPix cells, embedded by a small MLP, and every forecast-engine block's
pre-norm is shifted and scaled per token by it (DiT-style, zero-initialised). It never calls the
encoder, so a forcing stream need not be an encoder input at all.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from weathergen.common.config import Config
from weathergen.datasets.utils import coords_to_hpyidxs
from weathergen.model.norms import ZeroLinear

if TYPE_CHECKING:
    from weathergen.model.forcing import ForcingInput
    from weathergen.model.model import Model, ModelParams

DEFAULT_METHOD = "cross_attention"
# grids per stream are fixed, so a handful of entries covers every stream and a spoof
_CELL_IDX_CACHE_SIZE = 64

_REGISTRY: dict[str, type[ForecastConditioning]] = {}


def register(cls: type[ForecastConditioning]) -> type[ForecastConditioning]:
    """Class decorator making an implementation selectable by `forecast_conditioning.type`."""
    assert cls.method not in _REGISTRY, f"Duplicate forecast conditioning '{cls.method}'."
    _REGISTRY[cls.method] = cls
    return cls


def conditioning_method(cf: Config) -> str:
    """The forecast conditioning method a config selects."""
    return (cf.get("forecast_conditioning", None) or {}).get("type", DEFAULT_METHOD)


def dynamic_forcing_streams(cf: Config) -> list[str]:
    """Names of the streams marked `is_dynamic_forcing`, in config order."""
    return [name for name, info in cf.streams.items() if info.get("is_dynamic_forcing", False)]


class ForecastConditioning(nn.Module, ABC):
    """
    How a dynamic forcing conditions the forecast, once per rollout step.

    `ForcedModel.forward` calls, per step and before the forecast engine advances the latent,

        encoded = conditioning.encode(model, model_params, forcings, sampling_idxs, masks, device)
        tokens, fe_aux = conditioning(tokens, encoded)
        tokens = model.forecast_engine(tokens, step, coords, aux=fe_aux)

    `fe_aux` is None unless the implementation sets `fe_modulation_dim`, in which case the
    forecast engine is built with norms modulated by a conditioning of that width.
    """

    # value of `forecast_conditioning.type` selecting the implementation
    method: ClassVar[str]
    # matches state-dict keys only this method creates; used to refuse mixing methods on load
    state_key_pattern: ClassVar[re.Pattern]
    # not None -> the forecast engine's norms are modulated by a conditioning of this width
    fe_modulation_dim: int | None = None

    @classmethod
    @abstractmethod
    def from_config(
        cls, cf: Config, forcing_num_channels: dict[str, int] | None
    ) -> ForecastConditioning:
        """Build from the model config and the source channel count of each forcing stream."""

    def check_model(self, model: Model) -> None:
        """Raise if the built model cannot carry this conditioning. Called after `create`."""
        if self.fe_modulation_dim is not None and not self._modulations(model):
            raise ValueError(
                f"forecast_conditioning.type: {self.method} modulates the forecast engine's "
                "norms, but the forecast engine has no blocks (fe_num_blocks: 0 or no "
                "forecast.policy), so the conditioning would have no effect."
            )

    def _modulations(self, model: Model) -> list[ZeroLinear]:
        """The forecast engine's modulations, driven by this conditioning's `aux`."""
        return [m for m in model.forecast_engine.modules() if isinstance(m, ZeroLinear)]

    def is_active(self, forcings: ForcingInput) -> bool:
        """Whether a rollout step is conditioned at all; by default, when there is a forcing."""
        return not forcings.is_empty

    @abstractmethod
    def encode(
        self,
        model: Model,
        model_params: ModelParams,
        forcings: ForcingInput,
        sampling_idxs: list[int],
        source_masks: list[dict[str, Any]],
        device: torch.device,
    ) -> Any:
        """Read and embed the forcing of one rollout step, one sampling index per sample."""

    @abstractmethod
    def forward(self, tokens: torch.Tensor, encoded: Any) -> tuple[torch.Tensor, Any]:
        """Apply an encoded forcing: return the tokens and the forecast engine's `aux`."""

    @abstractmethod
    def fsdp_modules(self) -> Iterable[nn.Module]:
        """Modules offered to `fully_shard`, filtered by type in `init_model_and_shard`."""

    @abstractmethod
    def conditioning_parameters(self, model: Model) -> Iterator[nn.Parameter]:
        """Every parameter of this conditioning, including those living in the forecast engine."""

    def untrained_reason(self, model: Model) -> str | None:
        """Why the weights look untrained, or None when they do not or it cannot be told."""
        return None


def build_forecast_conditioning(
    cf: Config, forcing_num_channels: dict[str, int] | None = None
) -> ForecastConditioning:
    """The forecast conditioning `cf` selects (see `forecast_conditioning.type`)."""
    # imported for its registration: forcing.py defines the cross-attention engine
    import weathergen.model.forcing  # noqa: F401

    method = conditioning_method(cf)
    if method not in _REGISTRY:
        raise ValueError(
            f"Unknown forecast_conditioning.type '{method}', expected one of {sorted(_REGISTRY)}."
        )
    return _REGISTRY[method].from_config(cf, forcing_num_channels)


def check_conditioning_keys(method: str, keys: Iterable[str]) -> None:
    """
    Refuse checkpoint keys that belong to a forecast conditioning other than `method`.

    Cross-attention and AdaLN conditioning are separate model families: a checkpoint of one is
    never loaded, continued or finetuned as the other, and `load_state_dict(strict=False)`
    would otherwise drop the foreign weights with only a warning.
    """
    # imported for its registration: forcing.py defines the cross-attention engine
    import weathergen.model.forcing  # noqa: F401

    keys = list(keys)
    for other, cls in _REGISTRY.items():
        if other == method:
            continue
        foreign = [k for k in keys if cls.state_key_pattern.search(k)]
        if foreign:
            raise ValueError(
                f"The checkpoint was trained with forecast conditioning '{other}' ("
                f"{len(foreign)} weights, e.g. '{foreign[0]}'), but the model is configured "
                f"with '{method}'. The two are not interchangeable; configure "
                f"forecast_conditioning.type: {other} or start from a '{method}' checkpoint."
            )


####################################################################################################
# pooling of a forcing point cloud onto the latent HEALPix cells


def forcing_cell_index(coords: NDArray, healpix_level: int) -> NDArray:
    """
    Nested HEALPix cell of every (lat, lon) point, in the tokenizer's convention.

    Same mapping as `tokenizer_utils.hpy_cell_splits`, which decides the cell a source point is
    tokenized into, so cell `i` here is latent cell `i`.
    """
    coords = np.asarray(coords)
    return np.asarray(coords_to_hpyidxs(healpix_level, coords[:, 0], coords[:, 1]), dtype=np.int64)


def scatter_to_cells(
    values: torch.Tensor, cell_idx: torch.Tensor, num_cells: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scatter-mean per-point values onto cells, ignoring non-finite values per channel.

    values : ``(num_points, num_vars)``; cell_idx : ``(num_points,)``.

    Returns ``(cell_values, cell_valid)``, both ``(num_cells, num_vars)``: the per-cell mean
    (0 where nothing valid fell into a cell) and the fraction of a cell's points that were valid.
    """
    num_points, num_vars = values.shape
    idx = cell_idx.view(num_points, 1).expand(num_points, num_vars)

    valid = torch.isfinite(values)
    filled = torch.where(valid, values, torch.zeros_like(values))
    zeros = torch.zeros(num_cells, num_vars, device=values.device, dtype=values.dtype)

    sums = zeros.scatter_add(0, idx, filled)
    counts = zeros.scatter_add(0, idx, valid.to(values.dtype))
    total = zeros.scatter_add(0, idx, torch.ones_like(filled))

    cell_values = torch.where(counts > 0, sums / counts.clamp_min(1.0), zeros)
    cell_valid = torch.where(total > 0, counts / total.clamp_min(1.0), zeros)
    return cell_values, cell_valid


class LearnedForcingPool(nn.Module):
    """
    Learned replacement for the scatter-mean: a per-point, per-channel score normalised by a
    segment-softmax within each cell, so cells with more points just have more softmax terms.
    Invalid points get zero weight, independently per channel.
    """

    def __init__(self, num_vars: int, hidden_factor: int = 4) -> None:
        super().__init__()
        hidden = max(hidden_factor * num_vars, 8)
        self.score = nn.Sequential(
            nn.Linear(num_vars, hidden), nn.SiLU(), nn.Linear(hidden, num_vars)
        )

    def forward(
        self, values: torch.Tensor, cell_idx: torch.Tensor, num_cells: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_points, num_vars = values.shape
        idx = cell_idx.view(num_points, 1).expand(num_points, num_vars)

        valid = torch.isfinite(values)
        filled = torch.where(valid, values, torch.zeros_like(values))
        zeros = torch.zeros(num_cells, num_vars, device=values.device, dtype=values.dtype)

        logits = self.score(filled).to(values.dtype).masked_fill(~valid, -1e9)
        exp_logits = torch.exp(logits - logits.detach().amax(dim=0, keepdim=True))
        exp_logits = exp_logits * valid.to(values.dtype)
        denom = zeros.scatter_add(0, idx, exp_logits)
        weights = exp_logits / denom.gather(0, idx).clamp_min(1e-12)
        cell_values = zeros.scatter_add(0, idx, weights * filled)

        counts = zeros.scatter_add(0, idx, valid.to(values.dtype))
        total = zeros.scatter_add(0, idx, torch.ones_like(filled))
        cell_valid = torch.where(total > 0, counts / total.clamp_min(1.0), zeros)
        cell_values = torch.where(cell_valid > 0, cell_values, zeros)
        return cell_values, cell_valid


class ForcingEmbed(nn.Module):
    """
    Per-cell forcing embedding: [values | valid fraction] -> MLP -> conditioning width.

    The forcing arrives normalised by its reader's statistics (`ForcingInput`), so unlike the
    original this holds no mean/stdev buffers. Values of cells without data are zeroed and the
    valid-fraction feature tells the model they are absent.
    """

    def __init__(self, num_vars: int, dim_cond: int, hidden_factor: int = 2) -> None:
        super().__init__()
        self.name = "forecast_conditioning_embed"
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_vars, hidden_factor * dim_cond),
            nn.SiLU(),
            nn.Linear(hidden_factor * dim_cond, dim_cond),
        )

    def forward(self, cell_values: torch.Tensor, cell_valid: torch.Tensor) -> torch.Tensor:
        cell_values = torch.where(cell_valid > 0, cell_values, torch.zeros_like(cell_values))
        x = torch.cat([cell_values, cell_valid], dim=-1)
        return self.mlp(x.to(self.mlp[0].weight.dtype))


####################################################################################################
# time conditioning


def broadcast_time(time: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """``[B, dim]`` -> ``[B, num_tokens, dim]``: time is global, every token sees it."""
    return time.unsqueeze(1).expand(time.shape[0], num_tokens, time.shape[-1])


# period of each time variable, in the unit its raw value is given in
TIME_PERIODS = {"start_day": 365.0, "end_day": 365.0, "start_time": 24.0, "end_time": 24.0}
TIME_TRANSFORMS = ("absolute", "absolute_normalized", "cos_sin", "fourier")


def _day_of_year(dt: np.datetime64) -> float:
    """1-indexed day of year."""
    jan1 = dt.astype("datetime64[Y]").astype("datetime64[D]")
    return float((dt.astype("datetime64[D]") - jan1).astype(int) + 1)


def _hour_of_day(dt: np.datetime64) -> float:
    """Fractional hour of day in [0, 24)."""
    return float((dt - dt.astype("datetime64[D]")) / np.timedelta64(1, "h"))


def encode_time_window(
    start: np.datetime64,
    end: np.datetime64,
    variables: list[str],
    transform: str,
    emb_dimension: int,
) -> list[float]:
    """
    Time features of a window, as `DataReaderCondition._encode` computes them.

    Days are fractional (day 25 at 12:00 is 25.5), so every 6 h window is distinct. `fourier`
    gives, per variable, cos/sin at the first `emb_dimension // 2` harmonics of its period.
    """
    raw = {
        "start_day": _day_of_year(start) + _hour_of_day(start) / 24.0,
        "start_time": _hour_of_day(start),
        "end_day": _day_of_year(end) + _hour_of_day(end) / 24.0,
        "end_time": _hour_of_day(end),
    }
    values: list[float] = []
    for var in variables:
        if transform == "absolute":
            values.append(raw[var])
        elif transform == "absolute_normalized":
            values.append(raw[var] / TIME_PERIODS[var])
        elif transform == "cos_sin":
            angle = 2.0 * np.pi * raw[var] / TIME_PERIODS[var]
            values += [np.cos(angle), np.sin(angle)]
        elif transform == "fourier":
            for k in range(1, emb_dimension // 2 + 1):
                angle = 2.0 * np.pi * k * raw[var] / TIME_PERIODS[var]
                values += [np.cos(angle), np.sin(angle)]
    return [float(v) for v in values]


class TimeConditioning(nn.Module):
    """
    The valid time of a forecast step as a conditioning vector, one per sample.

    A port of the `condition` stream of `ForcingInjection` (`DataReaderCondition`): the same
    features of the same window -- the one the step predicts, on the component's own, unlagged
    timeline -- but computed from the window directly, without a reader or a sampler path, and
    per sample instead of shared by the batch. An MLP maps them to the conditioning width.
    """

    def __init__(
        self,
        dim_cond: int,
        variables: list[str] | None = None,
        transform: str = "fourier",
        emb_dimension: int = 16,
        hidden_factor: int = 2,
    ) -> None:
        super().__init__()
        self.name = "forecast_conditioning_time"
        self.variables = list(variables or ["start_day", "start_time", "end_day", "end_time"])
        unknown = set(self.variables) - set(TIME_PERIODS)
        if unknown:
            raise ValueError(f"Unknown time variables {sorted(unknown)}, known: {TIME_PERIODS}.")
        if transform not in TIME_TRANSFORMS:
            raise ValueError(f"Unknown time transform '{transform}', expected {TIME_TRANSFORMS}.")
        if transform == "fourier" and emb_dimension < 2:
            raise ValueError("The fourier time transform needs emb_dimension >= 2.")
        self.transform = transform
        self.emb_dimension = emb_dimension

        per_variable = {
            "absolute": 1,
            "absolute_normalized": 1,
            "cos_sin": 2,
            "fourier": 2 * (emb_dimension // 2),
        }[transform]
        self.num_features = per_variable * len(self.variables)
        self.mlp = nn.Sequential(
            nn.Linear(self.num_features, hidden_factor * dim_cond),
            nn.SiLU(),
            nn.Linear(hidden_factor * dim_cond, dim_cond),
        )

    @classmethod
    def from_config(cls, time_cfg, dim_cond: int, hidden_factor: int) -> TimeConditioning | None:
        """The time conditioning a `forecast_conditioning.time` block asks for, or None."""
        if not time_cfg:
            return None
        return cls(
            dim_cond,
            variables=time_cfg.get("variables", None),
            transform=time_cfg.get("transform", "fourier"),
            emb_dimension=time_cfg.get("emb_dimension", 16),
            hidden_factor=hidden_factor,
        )

    def features(self, forcings: ForcingInput, sampling_idxs: list[int]) -> NDArray:
        """``[B, num_features]`` for the windows the samples' current step predicts."""
        windows = [forcings.time_window_handler.window(idx) for idx in sampling_idxs]
        return np.asarray(
            [
                encode_time_window(
                    w.start, w.end, self.variables, self.transform, self.emb_dimension
                )
                for w in windows
            ],
            dtype=np.float32,
        )

    def forward(self, forcings: ForcingInput, sampling_idxs: list[int], device) -> torch.Tensor:
        feats = torch.as_tensor(self.features(forcings, sampling_idxs), device=device)
        return self.mlp(feats.to(self.mlp[0].weight.dtype))


@register
class AdaLNConditioning(ForecastConditioning):
    """
    Per-cell AdaLN forecast conditioning, a port of `ForcingInjection(mode="adaln_local")`.

    Per step and sample, every dynamic forcing stream is read through `ForcingInput.get_fields`
    (lag, coupling and elevation included), pooled onto this model's HEALPix cells, the streams'
    channels concatenated, and embedded to `dim_cond`. The result is broadcast to the forecast
    engine's token layout -- zeros for the register/class tokens, repeated over the queries of a
    cell -- and every forecast-engine block shifts and scales its normed input by it
    (`ModulatedLayerNorm`). The modulation is a `ZeroLinear`, so the conditioned model starts
    as the unconditioned one on every initialisation path and aux tokens are never modulated.

    With a `time` block, the step's valid time (`TimeConditioning`, the original's `condition`
    stream) is added to the conditioning of every token, aux tokens included: time is global.
    Time alone, without any dynamic forcing stream, is a valid configuration.

    Differences to the original: the forcing comes through `ForcingInput` instead of a dedicated
    `type: forcing` sampler path, it is per sample instead of shared by the batch, time is derived
    from the step's window instead of read from a `condition` stream and is optional, and the FE
    keeps its own norms, with the modulation added on top.
    """

    method = "adaln"
    # only the forcing path: the time MLP and the FE modulation are shared with cross-attention
    state_key_pattern = re.compile(r"forcing_engine\.(embed|pools)\.")

    def __init__(
        self,
        streams: list[str],
        stream_channels: list[int],
        healpix_level: int,
        num_queries: int,
        dim_cond: int = 256,
        hidden_factor: int = 2,
        pool: str = "mean",
        time: dict | None = None,
    ) -> None:
        super().__init__()
        assert len(streams) == len(stream_channels)
        if pool not in ("mean", "learned"):
            raise ValueError(f"forecast_conditioning.pool must be mean or learned, got '{pool}'.")

        self.streams = list(streams)
        self.stream_channels = list(stream_channels)
        self.healpix_level = healpix_level
        self.num_cells = 12 * 4**healpix_level
        self.num_queries = num_queries
        self.fe_modulation_dim = dim_cond

        self.pools = (
            nn.ModuleDict(
                {
                    _module_key(s): LearnedForcingPool(c)
                    for s, c in zip(self.streams, self.stream_channels, strict=True)
                }
            )
            if pool == "learned"
            else None
        )
        self.embed = (
            ForcingEmbed(sum(self.stream_channels), dim_cond, hidden_factor)
            if self.streams
            else None
        )
        self.time = TimeConditioning.from_config(time, dim_cond, hidden_factor)
        if self.embed is None and self.time is None:
            raise ValueError(
                "forecast_conditioning.type: adaln needs a stream with is_dynamic_forcing: true, "
                "a forecast_conditioning.time block, or both."
            )

        # the point -> cell index of a forcing grid, keyed by stream and grid; grids are fixed
        # per stream, so this is computed once per stream and device
        self._cell_idx_cache: dict[tuple, torch.Tensor] = {}

    @classmethod
    def from_config(
        cls, cf: Config, forcing_num_channels: dict[str, int] | None
    ) -> AdaLNConditioning:
        streams = dynamic_forcing_streams(cf)
        if streams and (
            forcing_num_channels is None or not set(streams) <= set(forcing_num_channels)
        ):
            raise ValueError(
                f"AdaLN conditioning needs the source channel count of forcing streams {streams}, "
                f"got {forcing_num_channels}."
            )
        stream_channels = [forcing_num_channels[s] for s in streams]
        if any(c == 0 for c in stream_channels):
            raise ValueError(
                f"Every dynamic forcing stream needs source channels, got "
                f"{dict(zip(streams, stream_channels, strict=True))}."
            )

        ccf = cf.get("forecast_conditioning", None) or {}
        return cls(
            streams,
            stream_channels,
            healpix_level=cf.healpix_level,
            num_queries=cf.ae_local_num_queries,
            dim_cond=ccf.get("dim_cond", 256),
            hidden_factor=ccf.get("hidden_factor", 2),
            pool=ccf.get("pool", "mean"),
            time=ccf.get("time", None),
        )

    def _cell_idx(self, stream: str, coords: NDArray, device) -> torch.Tensor:
        coords = np.ascontiguousarray(coords)
        # keyed by the grid itself, so a different grid (e.g. a spoof) gets its own entry
        key = (stream, str(device), coords.shape, hash(coords.tobytes()))
        cached = self._cell_idx_cache.get(key)
        if cached is None:
            idx = forcing_cell_index(coords, self.healpix_level)
            cached = torch.as_tensor(idx, dtype=torch.long, device=device)
            if len(self._cell_idx_cache) >= _CELL_IDX_CACHE_SIZE:
                self._cell_idx_cache.clear()
            self._cell_idx_cache[key] = cached
        return cached

    def pool_fields(self, fields: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
        """``[B, num_cells, 2 * sum(channels)]`` = [values | valid] from `get_fields` output."""
        pooled = []
        for sample in fields:
            assert set(sample) == set(self.streams), (
                f"Forcing streams {sorted(sample)} do not match the conditioning's {self.streams}."
            )
            cell_values, cell_valid = [], []
            for stream, num_channels in zip(self.streams, self.stream_channels, strict=True):
                rdata = sample[stream]
                values = torch.as_tensor(np.asarray(rdata.data), dtype=torch.float32, device=device)
                assert values.ndim == 2 and values.shape[1] == num_channels, (
                    f"Forcing stream '{stream}' delivered data of shape {tuple(values.shape)}, "
                    f"expected {num_channels} channels."
                )
                if rdata.is_spoof:
                    # nothing was read: the stream is absent for this sample
                    values = torch.full_like(values, float("nan"))
                cell_idx = self._cell_idx(stream, rdata.coords, device)
                if self.pools is not None:
                    v, m = self.pools[_module_key(stream)](values, cell_idx, self.num_cells)
                else:
                    v, m = scatter_to_cells(values, cell_idx, self.num_cells)
                cell_values.append(v)
                cell_valid.append(m)
            pooled.append(torch.cat([*cell_values, *cell_valid], dim=-1))
        return torch.stack(pooled)

    def is_active(self, forcings) -> bool:
        return not forcings.is_empty or self.time is not None

    def encode(self, model, model_params, forcings, sampling_idxs, source_masks, device):
        """``(per-cell forcing [B, cells, dim_cond] | None, time [B, dim_cond] | None)``."""
        cells = None
        if self.embed is not None:
            fields = forcings.get_fields(sampling_idxs)
            pooled = self.pool_fields(fields, device)
            num_vars = pooled.shape[-1] // 2
            cells = self.embed(pooled[..., :num_vars], pooled[..., num_vars:])
        time = self.time(forcings, sampling_idxs, device) if self.time is not None else None
        return cells, time

    def forward(self, tokens: torch.Tensor, encoded: tuple):
        """Leave the tokens alone; return the per-token conditioning of the forecast engine."""
        cells, time = encoded
        batch_size, num_tokens, _ = tokens.shape
        num_cell_tokens = self.num_cells * self.num_queries
        num_aux = num_tokens - num_cell_tokens
        assert num_aux >= 0, f"{num_tokens} latent tokens for {num_cell_tokens} cell tokens."

        cond = None
        if cells is not None:
            assert cells.shape[:2] == (batch_size, self.num_cells), (
                f"Conditioning of shape {tuple(cells.shape)} for {batch_size} samples and "
                f"{self.num_cells} cells."
            )
            cond_cells = cells.repeat_interleave(self.num_queries, dim=1)
            cond_aux = cond_cells.new_zeros(batch_size, num_aux, cond_cells.shape[-1])
            cond = torch.cat([cond_aux, cond_cells], dim=1)
        if time is not None:
            # global: every token, register/class tokens included, sees the step's time
            time = broadcast_time(time, num_tokens)
            cond = time if cond is None else cond + time
        return tokens, cond

    def fsdp_modules(self) -> Iterable[nn.Module]:
        modules = [m for m in (self.embed, self.pools, self.time) if m is not None]
        return (m for module in modules for m in module.modules())

    def conditioning_parameters(self, model: Model) -> Iterator[nn.Parameter]:
        yield from self.parameters()
        for modulation in self._modulations(model):
            yield from modulation.parameters()

    def untrained_reason(self, model: Model) -> str | None:
        nonzero = 0
        for modulation in self._modulations(model):
            weight = modulation.weight.detach()
            to_local = getattr(weight, "to_local", None)
            weight = to_local() if to_local is not None else weight
            if weight.numel() == 0:
                # an empty shard on this rank cannot tell
                return None
            nonzero += int(torch.count_nonzero(weight).item())
        if nonzero == 0:
            return (
                "every forecast-engine modulation weight is exactly zero, its initialization, so "
                "the forcing does not reach the forecast"
            )
        return None


def _module_key(stream: str) -> str:
    """A stream name usable as a ModuleDict key."""
    return stream.replace(".", "_")
