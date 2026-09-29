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
from torch.distributed.fsdp import fully_shard
from torch.utils.checkpoint import checkpoint

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
    """The encoder in an outer checkpoint, re-run on the forcing at every step."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(DIM, DIM)
        self.engine = GlobalEngine()
        self.forcing = nn.Linear(DIM, DIM)
        self.forecast = nn.Linear(DIM, DIM)

    def encode(self, x):
        return checkpoint(self.engine, self.embed(x), use_reentrant=False)

    def forward(self, source, forcings):
        latent = self.encode(source)
        loss = 0
        for step in range(STEPS):
            latent = latent + checkpoint(
                self.forcing, self.encode(forcings[step]), use_reentrant=False
            )
            latent = checkpoint(self.forecast, latent, use_reentrant=False)
            loss = loss + latent.pow(2).mean()
        return loss


def _sharded(reference: nn.Module) -> nn.Module:
    """Shard like `init_model_and_shard`: blocks, then the rest, then the root."""
    model = copy.deepcopy(reference)
    for block in model.engine.blocks:
        fully_shard(block)
    fully_shard(model.forcing)
    fully_shard(model.forecast, reshard_after_forward=False)
    fully_shard(model)
    return model


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


def test_nested_recompute_of_a_module_called_per_step(process_group):
    torch.manual_seed(0)
    reference = RolloutWithForcingEncodes().cuda()
    source = torch.randn(BATCH, DIM, device="cuda")
    forcings = torch.randn(STEPS, BATCH, DIM, device="cuda")
    reference(source, forcings).backward()

    # plain FSDP2 reruns the blocks on their sharded weights in the outer recompute
    unpatched = _sharded(reference)
    with pytest.raises(RuntimeError, match="mixed torch.Tensor and DTensor"):
        unpatched(source, forcings).backward()

    model = _sharded(reference)
    unshard_in_nested_recompute(model)
    for _ in range(2):  # a second iteration must not trip over state left by the first
        model.zero_grad(set_to_none=True)
        model(source, forcings).backward()

    for (name, param), ref in zip(model.named_parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(param.grad.full_tensor(), ref.grad, msg=name)
