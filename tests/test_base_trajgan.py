#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from conv_gan.models.trajGAN import TrajGAN  # Assuming this is the correct import for your TrajGAN class


class MockModule(nn.Module):
    def __init__(self, param: float):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(float(param)))

# A concrete subclass of TrajGAN for testing purposes
class ConcreteTrajGAN(TrajGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt_g = torch.optim.Adam(self.gen.parameters())
        self.opt_d = torch.optim.Adam(self.dis.parameters())

    def get_noise(self, *args, **kwargs) -> torch.Tensor:
        return torch.randn(1)

    def forward(self, x) -> torch.Tensor:
        return x

    def training_loop(self, *args, **kwargs) -> None:
        pass

class TrajGANTest(unittest.TestCase):

    def setUp(self):
        self.gen = MockModule(0)
        self.dis = MockModule(1)
        # Create gen2 and dis2 with different parameters
        self.gen2 = MockModule(2)
        self.dis2 = MockModule(3)
        self.name = 'test_trajgan'
        self.temp_dir = tempfile.TemporaryDirectory()
        # Get path to temporary directory
        self.param_path = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization_without_dp(self):
        trajgan = ConcreteTrajGAN(name=self.name, gen=self.gen, dis=self.dis, dp=False, param_path=self.param_path)
        self.assertFalse(trajgan.dp)

    def test_initialization_with_dp(self):
        trajgan = ConcreteTrajGAN(name=self.name, gen=self.gen, dis=self.dis, dp=True, dp_in_dis=True, param_path=self.param_path)
        self.assertTrue(trajgan.dp)

    @patch('torch.save')
    def test_save_parameters_without_dp(self, mock_save):
        trajgan = ConcreteTrajGAN(name=self.name, gen=self.gen, dis=self.dis, dp=False, param_path=self.param_path)
        trajgan.save_parameters(epoch=1)
        self.assertEqual(mock_save.call_count, 3)

    @patch('torch.load')
    @patch('conv_gan.models.trajGAN.TrajGAN.load_state_dict')
    def test_load_parameters_without_dp(self, mock_load_state_dict, mock_load):
        trajgan = ConcreteTrajGAN(name=self.name, gen=self.gen, dis=self.dis, dp=False, param_path=self.param_path)
        trajgan.load_parameters(epoch=1)
        self.assertEqual(mock_load.call_count, 1)
        self.assertEqual(mock_load_state_dict.call_count, 1)

    def test_save_and_load_parameters_without_dp(self):
        trajgan = ConcreteTrajGAN(name=self.name, gen=self.gen, dis=self.dis, dp=False, dp_in_dis=False,
                                  param_path=self.param_path)
        # Verify that the parameter files don't exist yet
        paths = [trajgan.gen_weight_path.format(epoch=1),
                 trajgan.dis_weight_path.format(epoch=1),
                 trajgan.com_weight_path.format(epoch=1),]
        for path in paths:
            self.assertFalse(Path(path).exists(), msg=f"Path {path} already exists.")

        trajgan.save_parameters(epoch=1)
        # Verify that parameter files were created
        for path in paths:
            self.assertTrue(Path(path).exists(), msg=f"Path {path} does not exist.")

        loaded_trajgan = ConcreteTrajGAN(name=self.name, gen=self.gen2, dis=self.dis2, dp=False, dp_in_dis=False,
                                         param_path=self.param_path)
        # Verify that before loading, the parameters are not the same
        self.assertFalse(torch.equal(trajgan.gen.param, loaded_trajgan.gen.param),
                         msg="Gen Parameters are the same before loading.")
        self.assertFalse(torch.equal(trajgan.dis.param, loaded_trajgan.dis.param),
                         msg="Dis Parameters are the same before loading.")

        loaded_trajgan.load_parameters(epoch=1)
        # Verify that the loaded parameters are the same as the saved parameters
        self.assertTrue(torch.equal(trajgan.gen.param, loaded_trajgan.gen.param),
                        msg="Gen Parameters are not the same after loading.")
        self.assertTrue(torch.equal(trajgan.dis.param, loaded_trajgan.dis.param),
                        msg="Dis Parameters are not the same after loading.")

    def test_save_and_load_parameters_with_dp(self):
        trajgan = ConcreteTrajGAN(name=self.name, gen=self.gen, dis=self.dis, dp=True, dp_in_dis=True, param_path=self.param_path)
        # Verify that the parameter files don't exist yet
        paths = [trajgan.gen_weight_path.format(epoch=1), trajgan.dis_weight_path.format(epoch=1)]
        for path in paths:
            self.assertFalse(Path(path).exists(), msg=f"Path {path} already exists.")
        dl = DataLoader(torch.randn(10, 1, 28, 28))
        trajgan.init_dp(dataloader=dl, epochs=10, max_grad_norm=0.1, target_epsilon=10)
        trajgan.save_parameters(epoch=1)
        # Verify that parameter files were created
        for path in paths:
            self.assertTrue(Path(path).exists(), msg=f"Path {path} does not exist.")
        loaded_trajgan = ConcreteTrajGAN(name=self.name, gen=self.gen2, dis=self.dis2, dp=True, dp_in_dis=True, param_path=self.param_path)
        # Verify that before loading, the parameters are not the same
        self.assertFalse(torch.equal(trajgan.gen.param, loaded_trajgan.gen.param), msg="Gen Parameters are the same before loading.")
        self.assertFalse(torch.equal(trajgan.dis.param, loaded_trajgan.dis.param), msg="Dis Parameters are the same before loading.")
        loaded_trajgan.load_parameters(epoch=1, dataloader=dl)
        # Verify that the loaded parameters are the same as the saved parameters
        self.assertTrue(torch.equal(trajgan.gen.param, loaded_trajgan.gen.param), msg="Gen Parameters are not the same after loading.")
        self.assertTrue(torch.equal(trajgan.dis.param, loaded_trajgan.dis.param), msg="Dis Parameters are not the same after loading.")
        # Assert that the DP parameters are the same
        self.assertEqual(trajgan.dp_initialised, loaded_trajgan.dp_initialised)
        self.assertEqual(trajgan.dp_in_dis, loaded_trajgan.dp_in_dis)
        self.assertEqual(trajgan._target_epsilon, loaded_trajgan._target_epsilon)
        self.assertEqual(trajgan.dp_opt.max_grad_norm, loaded_trajgan.dp_opt.max_grad_norm)
        self.assertEqual(trajgan.dp_opt.noise_multiplier, loaded_trajgan.dp_opt.noise_multiplier)
        self.assertEqual(trajgan.delta, loaded_trajgan.delta)

if __name__ == '__main__':
    unittest.main()
