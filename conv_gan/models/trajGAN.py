#!/usr/bin/env python3
"""Base module for Trajectory Generators."""
import json
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List

import torch
from opacus import PrivacyEngine
from opacus.optimizers import DPOptimizer
from torch import nn
from torch.utils.data import DataLoader

from conv_gan.models.utils import prepare_param_path
from conv_gan.utils.helpers import compute_delta

log = logging.getLogger()


class TrajGAN(nn.Module, ABC):

    def __init__(
            self,
            name: str,
            gen: nn.Module,
            dis: nn.Module,
            dp: bool = False,
            dp_in_dis: bool = False,
            dp_accountant: str = 'prv',
            param_path: Optional[str] = None
    ):
        """
        Needs to be called at the start of the subclass' __init__ method.
        :param name: Name of the model
        :param dp: Whether to use differential privacy
        :param dp_in_dis: Whether to apply DP to the discriminator instead of the generator
        :param dp_accountant: Which accountant to use for DP-SGD
        :param param_path: Path where to store the model parameters
        """
        super().__init__()

        # Verify that self.gen and self.dis are defined
        self.gen = gen
        self.dis = dis

        # Store parameters
        self.name = name
        self.dp = dp
        self.dp_in_dis = dp_in_dis  # Determines if DP is applied to the discriminator [True] or the generator [False]

        # Set the parameter path
        self.prepare_param_paths(name=name, param_path=param_path)

        #######################################################################
        # Differential Private Stochastic Gradient Descent
        # Validate if model can be used for DP-SGD without modifications
        if self.dp:
            from opacus.validators import ModuleValidator
            errors = ModuleValidator.validate(self, strict=False)
            if len(errors) == 0:
                log.info("No errors found - model can be trained with DP.")
            else:
                log.error("The following errors prevent the model from DP training:")
                for e in errors:
                    log.error(str(e))
                    ModuleValidator.validate(self, strict=True)
            self.accountant = dp_accountant
            self.dp_initialised = False
            with warnings.catch_warnings(record=True) as w:
                self._privacy_engine = PrivacyEngine(accountant=self.accountant)
                if len(w) > 0:
                    for warning in w:
                        log.warning(str(warning.message))

    @property
    def dp_module(self) -> nn.Module:
        if self.dp_in_dis:
            return self.dis
        else:
            return self.gen

    @dp_module.setter
    def dp_module(self, module: nn.Module):
        if self.dp_in_dis:
            self.dis = module
        else:
            self.gen = module

    @property
    def dp_param_path(self) -> str:
        if self.dp_in_dis:
            return self.dis_weight_path
        else:
            return self.gen_weight_path

    @property
    def dp_opt(self) -> DPOptimizer:
        if self.dp_in_dis:
            return self.opt_d
        else:
            return self.opt_g

    @dp_opt.setter
    def dp_opt(self, optimizer: DPOptimizer):
        if self.dp_in_dis:
            self.opt_d = optimizer
        else:
            self.opt_g = optimizer

    @property
    def std_module(self) -> nn.Module:
        if self.dp_in_dis:
            return self.gen
        else:
            return self.dis

    @property
    def std_param_path(self) -> str:
        if self.dp_in_dis:
            return self.gen_weight_path
        else:
            return self.dis_weight_path

    @property
    def device(self):
        """Assumes all parameters are on the same device."""
        return next(self.parameters()).device

    @abstractmethod
    def get_noise(self, *args, **kwargs) -> torch.Tensor:
        """Returns suitable noise for the generator's forward method. Has to be implemented by all subclasses."""
        pass

    def generate(self, num: int, length: Optional[int] = None) -> torch.Tensor or List[torch.Tensor]:
        """Generate num samples of length length.

        :param num: Number of samples to generate
        :param length: Length of each sample. Not required for fixed-length models.
        :return: None
        """
        return self.forward(self.get_noise(batch_size=num, num_time_steps=length))

    def forward(self, x) -> torch.Tensor or List[torch.Tensor]:
        """Has to be implemented by all subclasses."""
        return self.gen(x)

    @abstractmethod
    def training_loop(self, *args, **kwargs) -> None:
        """Has to be implemented by all subclasses."""
        pass

    # Create properties for delta and epsilon
    @property
    def delta(self) -> float:
        if self.dp:
            return self._delta
        else:
            raise RuntimeError("Model is trained without differential privacy.")

    @property
    def epsilon(self) -> float:
        if self.dp:
            return self._privacy_engine.get_epsilon(self.delta)
        else:
            raise RuntimeError("Model is trained without differential privacy.")

    def prepare_param_paths(self, name: str, param_path: Optional[str] = None):
        self.param_path = prepare_param_path(name=name, param_path=param_path)
        self.com_weight_path = self.param_path.format(EPOCH="{epoch:04d}", MODEL="COM")
        self.gen_weight_path = self.param_path.format(EPOCH="{epoch:04d}", MODEL="GEN")
        self.dis_weight_path = self.param_path.format(EPOCH="{epoch:04d}", MODEL="DIS")
        # Get the directory containing the parameters as absolute path
        self.param_dir = str(Path(self.param_path).parent.absolute()) + '/'
        log.info(f"Saving parameters to {self.param_dir}")


    def save_parameters(self, epoch: int):
        # Create directory if it does not exist
        Path(self.com_weight_path).parent.mkdir(parents=True, exist_ok=True)
        if not self.dp:
            torch.save(self.gen.state_dict(), self.gen_weight_path.format(epoch=epoch))
            torch.save(self.dis.state_dict(), self.dis_weight_path.format(epoch=epoch))
            torch.save(self.state_dict(), self.com_weight_path.format(epoch=epoch))
            log.info(f"Saved parameters to {self.com_weight_path}")
        else:
            if not self.dp_initialised:
                raise RuntimeError("Checkpointing only possible after dp_init() has been called.")

            # Use privacy engine's checkpointing functionality
            # 1. Save non DP submodule as usual
            torch.save(self.std_module.state_dict(), self.std_param_path.format(epoch=epoch))
            # 2. Save generator via privacy engine
            self._privacy_engine.save_checkpoint(
                path=self.dp_param_path.format(epoch=epoch),
                module=self.dp_module,
                optimizer=self.dp_opt,
            )
            # 3. Save DP parameters
            dp_params = {
                'delta': self.delta,
                'max_grad_norm': self.dp_opt.max_grad_norm,
                'noise_multiplier': self.dp_opt.noise_multiplier,
                'epochs': self.epochs,
                'target_epsilon': self._target_epsilon
            }
            with open(self.param_dir + 'dp_params.json', 'w') as f:
                json.dump(dp_params, f)
            log.info(f"Saved parameters to {self.param_dir}.")

    def load_parameters(self, epoch: int, dataloader: DataLoader = None) -> DataLoader or None:
        
        if not self.dp:
            self.load_state_dict(torch.load(self.com_weight_path.format(epoch=epoch), map_location=self.device))
            print(f"Loaded parameters from {self.com_weight_path.format(epoch=epoch)}")
        else:
            if dataloader is None:
                raise RuntimeError("DataLoader has to be provided if loading DP model")
            # Load standard parameters
            self.std_module.load_state_dict(torch.load(self.std_param_path.format(epoch=epoch)))
            # Load dp parameters
            with open(self.param_dir + 'dp_params.json', 'r') as f:
                dp_params = json.load(f)
            self._delta = dp_params['delta']
            self.epochs = dp_params['epochs']
            self._target_epsilon = dp_params['target_epsilon']
            # Load privacy engine & generator
            self.dp_module, self.dp_opt, dataloader = self._privacy_engine.make_private(
                module=self.dp_module,
                optimizer=self.dp_opt,
                data_loader=dataloader,
                noise_multiplier=dp_params['noise_multiplier'],
                max_grad_norm=dp_params['max_grad_norm']
            )
            self._privacy_engine.load_checkpoint(
                path=self.dp_param_path.format(epoch=epoch),
                module=self.dp_module,
                optimizer=self.dp_opt
            )
            self.dp_initialised = True
            log.info(f"Loaded parameters from {self.param_path}")
            return dataloader

    def init_dp(self,
                dataloader: DataLoader,
                epochs: int,
                max_grad_norm: float,
                noise_multiplier: float = None,
                target_epsilon: float = None,
                delta: float = None,
                ):
        log.info("Initializing privacy engine, might take some time.")
        # The privacy engine raises a few warning we cannot do anything about
        warnings.simplefilter("ignore")
        self._delta = compute_delta(len(dataloader.dataset)) if delta is None else delta
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs

        if target_epsilon is not None:
            self._target_epsilon = target_epsilon
            log.info(f"Targeting (ε = {self._target_epsilon}, δ = {self.delta})")
        elif noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
            self._target_epsilon = 0  # Important for saving checkpoints
            log.info(f"Utilizing (σ = {self.noise_multiplier}, δ = {self.delta})")
        else:
            raise ValueError("Either target_epsilon or noise_multiplier have to be provided!")

        compatible = self._privacy_engine.is_compatible(
            module=self.dp_module,
            optimizer=self.dp_opt,
            data_loader=dataloader
        )
        if compatible:
            log.info("Model compatible with privacy settings!")
        else:
            raise RuntimeError("Model, Optimizer or dataset not compatible with DP-SGD.")

        if target_epsilon is not None:
            self.dp_module, self.dp_opt, dataloader = self._privacy_engine.make_private_with_epsilon(
                module=self.dp_module,
                optimizer=self.dp_opt,
                data_loader=dataloader,
                epochs=self.epochs,
                target_epsilon=self._target_epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=True,
                # grad_sample_mode="ew"  # functorch
            )
            warnings.simplefilter("default")
        else:
            self.dp_module, self.dp_opt, dataloader = self._privacy_engine.make_private(
                module=self.dp_module,
                optimizer=self.dp_opt,
                data_loader=dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=True
            )
        self.dp_initialised = True
        log.info(f"Using σ={self.dp_opt.noise_multiplier} and C={self.max_grad_norm}")
        return dataloader
