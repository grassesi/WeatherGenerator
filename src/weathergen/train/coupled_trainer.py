import logging

import torch
import tqdm

import weathergen.common.config as config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.model.model_interface import init_model_and_shard
from weathergen.train.loss_calculator import LossCalculator
from weathergen.train.trainer import Trainer
from weathergen.train.utils import TRAIN, VAL, TEST
from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)


class CoupledTrainer(Trainer):
    """Drives/owns the  loop."""

    def __init__(self, training_logging, components: dict[str, Trainer]):
        super().__init__(training_logging)
        
        self._components: dict[str, Trainer] = components
    
    def inference(self, devices, run_id_contd, mini_epoch_contd):
        # GOAL: handle as much as possible in component Trainer instances,
        # do instantiation for anything dataset/sampling/communication related here
        # => pass it to components
        
        for name, component in self._components.items():
            component.inference(cf, devices, run_id_contd, mini_epoch_contd)
        
        # TODO: what is the config for the driver?
        # => just select one of the components configs?
        self.init(None, devices)
        
        device_type = torch.accelerator.current_accelerator()
        self.device = torch.device(f"{device_type}:{cf.local_rank}")
        
        # TODO take care that batches contain data for all components
        self.dataset = MultiStreamDataSampler(
            self.cf,
            self.test_cfg,
            stage=VAL
        )
        self.dataset_val = self.dataset
        
        # TODO instatiate ForcingInputs
        
        # make sure number of loaders does not exceed requested samples
        loader_num_workers = min(self.test_cfg.samples_per_mini_epoch, cf.data_loading.num_workers)
        loader_params = {
            "batch_size": None,
            "batch_sampler": None,
            "shuffle": False,
            "num_workers": loader_num_workers,
            "pin_memory": cf.data_loading.get("memory_pinning", False),
            "persistent_workers": cf.data_loading.get("persistent_workers", False),
        }
        self.data_loader_validation = torch.utils.data.DataLoader(
            self.dataset, **loader_params, sampler=None
        )
        
        for component in self._components:
            component.model, component.model_params = init_model_and_shard(
                component.cf,
                run_id_contd, # TODO different per compoent
                mini_epoch_contd, # TODO different per compoent
                component.test_cfg.training_mode,
                devices[0],
                component.with_ddp, # TODO set it globaly instead?
                component.with_fsdp # TODO set it globaly instead?
            )
            component.inference(self, cf, devices, run_id_contd, mini_epoch_contd)
        
            # get target_aux calculators for different loss terms
            component.target_and_aux_calculators_val = component.get_target_aux_calculators(self.test_cfg)
            component.loss_calculator_val = LossCalculator(component.cf, self.test_cfg, VAL, device=self.devices[0])
            
            if is_root():
                config.save(component.cf, mini_epoch=0)
            
        self.validate(0, self.test_cfg, self.batch_size_test_per_gpu)
        logger.info(f"Finished coupled inference run")
    
    def validate(self, mini_epoch: int, mode_cfg: Config, batch_size: int):
        # TODO
        pass
        # set model(s) to eval mode
        with torch.no_grad()
    