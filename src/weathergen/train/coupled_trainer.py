import logging

import torch
import tqdm

import weathergen.common.config as config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.model.model_interface import init_model_and_shard
from weathergen.train.loss_calculator import LossCalculator
from weathergen.train.trainer import Trainer, TRAIN, VAL, TEST
from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)


class CoupledTrainer(Trainer):
    def __init__(self, training_logging: config.Config, streams: config.Config):
        super().__init__(training_logging)
        self.coupling_config = couplings
        self._models: dict[str, tuple[Model, ModelParams]] = {}
    
    def inference(self, devices, run_id_contd, mini_epoch_contd):
        # general initalization
        self.init(cf, devices)

        cf = self.cf
        device_type = torch.accelerator.current_accelerator()
        self.device = torch.device(f"{device_type}:{cf.local_rank}")
        self.ema_model = None

        self.data_loader_validation = self.get_dataloader(self, cf)
        for run_id_contd, mini_epoch_contd in self.couplings:
            # TODO try to fit models onto different devices?
            self.model, self.model_params = init_model_and_shard(
                cf,
                self.dataset,
                run_id_contd,
                mini_epoch_contd,
                self.test_cfg.training_mode,
                devices[0],
                cf.with_ddp,
                cf.with_fsdp,
            )

        # get target_aux calculators for different loss terms
        self.target_and_aux_calculators_val = self.get_target_aux_calculators(self.test_cfg)

        self.loss_calculator_val = LossCalculator(cf, self.test_cfg, VAL, device=self.devices[0])

        if is_root():
            config.save(self.cf, mini_epoch=0)

        logger.info(f"Starting inference with id={self.cf.general.run_id}.")

        # inference validation set
        self.validate(0, self.test_cfg, self.batch_size_test_per_gpu)
        logger.info(f"Finished inference run with id: {cf.general.run_id}")
    
    def inference2(self):
        
        self.model.eval()
        num_samples_write = mode_cfg.get("output", {}).get("num_samples", 0) * batch_size
        
        with (
            torch.no_grad(), 
            tqdm.tqdm(
                total=len(self.data_loader_validation), disable=self.cf.with_ddp
            ) as pbar,
            torch.autocast(
                device_type=f"cuda:{self.cf.local_rank}",
                dtype=self.mixed_precision_dtype,
                enabled=self.cf.with_mixed_precision,
            )
        ):
            
            for bidx, batch in enumerate(dataset_val_iter):
                if cf.data_loading.get("memory_pinning", False):
                    # pin memory for faster CPU-GPU transfer
                    batch = batch.pin_memory()

                batch.to_device(self.device)
                preds = self.model(
                    self.model_params,
                    batch.get_source_samples(),
                )
            
    
    def get_dataloader(self, cf):
        # NO DATALOADER needed, since during coupling batches can not be prefetched. => try to keep all data in device memory, use device-device transfer if needed.
        
        # (spoofed) Targets need to be generated
        
        # create data loader
        # only one needed since we only run the validation code path
        self.dataset = MultiStreamDataSampler(
            cf,
            self.test_cfg,
            stage=VAL,
        )
        self.dataset_val = self.dataset

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
        return torch.utils.data.DataLoader(
            self.dataset, **loader_params, sampler=None
        )