# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
GPU test of `unshard_in_nested_recompute`: an encoder with nested checkpoints around FSDP-sharded
blocks, called once for the source and once per rollout step as the cross-attention forcing does.

One process with a world size of 1 is enough: sharded parameters are DTensors regardless.
"""

import copy
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import CheckpointError, checkpoint

from weathergen.model.model_interface import unshard_in_nested_recompute

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP test needs a GPU")

DIM, BATCH, STEPS = 16, 4, 3


class GlobalEngine(nn.Module):
    """Checkpoints each block, like `GlobalAssimilationEngine.forward`."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, DIM)) for _ in range(3)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = x + checkpoint(block, x, use_reentrant=False)
        return x


class RolloutWithForcingEncodes(nn.Module):
    """
    The encoder in an outer checkpoint, re-run on the forcing at every step. Its tokens enter the
    sharded blocks in float32, as the real encoder's do, so a mixed-precision policy casts them.
    """

    def __init__(self, nested: bool = True):
        super().__init__()
        self.nested = nested
        self.embed = nn.Linear(DIM, DIM)
        self.engine = GlobalEngine()
        self.forcing = nn.Linear(DIM, DIM)
        self.forecast = nn.Linear(DIM, DIM)

    def encode(self, x):
        tokens = self.embed(x).float()
        if not self.nested:
            return self.engine(tokens)
        return checkpoint(self.engine, tokens, use_reentrant=False)

    def forward(self, source, forcings):
        latent = self.encode(source)
        loss = 0
        for step in range(STEPS):
            latent = latent + checkpoint(
                self.forcing, self.encode(forcings[step]), use_reentrant=False
            )
            latent = checkpoint(self.forecast, latent, use_reentrant=False)
            loss = loss + latent.float().pow(2).mean()
        return loss


def _sharded(model: nn.Module, mp_policy: MixedPrecisionPolicy | None) -> nn.Module:
    """Shard a copy like `init_model_and_shard`: blocks, then the rest, then the root."""
    model = copy.deepcopy(model)
    kwargs = {"mp_policy": mp_policy} if mp_policy is not None else {}
    for block in model.engine.blocks:
        fully_shard(block, **kwargs)
    fully_shard(model.forcing, **kwargs)
    fully_shard(model.forecast, reshard_after_forward=False, **kwargs)
    fully_shard(model, **kwargs)
    return model


def _grads(model: nn.Module, source, forcings, iterations: int = 1) -> list[torch.Tensor]:
    for _ in range(iterations):  # a second iteration must not trip over state left by the first
        model.zero_grad(set_to_none=True)
        model(source, forcings).backward()
    return [
        p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
        for p in model.parameters()
    ]


@pytest.fixture(scope="module")
def process_group():
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=str(port))
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "mp_policy",
    [None, MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)],
    ids=["fp32", "bf16"],
)
def test_nested_recompute_of_a_module_called_per_step(process_group, mp_policy):
    torch.manual_seed(0)
    model = RolloutWithForcingEncodes().cuda()
    source = torch.randn(BATCH, DIM, device="cuda")
    forcings = torch.randn(STEPS, BATCH, DIM, device="cuda")

    # plain FSDP2 reruns the blocks sharded (mixed Tensor/DTensor) and with uncast inputs
    # (recomputed float32 where the forward saved bfloat16)
    with pytest.raises((RuntimeError, CheckpointError)):
        _grads(_sharded(model, mp_policy), source, forcings)

    patched = _sharded(model, mp_policy)
    unshard_in_nested_recompute(patched)
    grads = _grads(patched, source, forcings, iterations=2)

    # the same sharding and precision without the outer checkpoint; it needs the hook too once the
    # float32 residual stream reaches a per-step module (`forcing`) under a mixed-precision policy
    flat = copy.deepcopy(model)
    flat.nested = False
    flat = _sharded(flat, mp_policy)
    unshard_in_nested_recompute(flat)
    expected = _grads(flat, source, forcings)
    for (name, _), grad, ref in zip(model.named_parameters(), grads, expected, strict=True):
        torch.testing.assert_close(grad, ref, msg=name)
