import unittest
import torch
import torch.nn as nn
from gpu_optimizer import GPUPowerOptimizer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 111 * 111, 10)

    def forward(self, x):
        x = self.pool(self.conv(x))
        x = x.view(-1, 64 * 111 * 111)
        return self.fc(x)

class TestGPUPowerOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = GPUPowerOptimizer()
        cls.model = SimpleModel().cuda()
        cls.sample_input = torch.randn(4, 3, 224, 224).cuda()

    def test_init(self):
        """Test if optimizer initializes correctly"""
        self.assertIsNotNone(self.optimizer)
        self.assertIsNotNone(self.optimizer.handle)
        self.assertIsNotNone(self.optimizer.initial_metrics)

    def test_get_current_metrics(self):
        """Test metrics collection"""
        metrics = self.optimizer._get_current_metrics()
        self.assertGreater(metrics.power_usage, 0)
        self.assertGreater(metrics.temperature, 0)
        self.assertGreater(metrics.clock_speed, 0)
        self.assertGreater(metrics.memory_clock, 0)

    def test_optimize_batch_size(self):
        """Test batch size optimization"""
        optimal_size = self.optimizer.optimize_batch_size(
            self.model,
            self.sample_input,
            target_power=200.0
        )
        self.assertGreater(optimal_size, 0)
        self.assertLessEqual(optimal_size, self.sample_input.shape[0])

    def test_suggest_power_efficient_config(self):
        """Test configuration suggestions"""
        config = self.optimizer.suggest_power_efficient_config()
        self.assertIn('recommended_batch_size_factor', config)
        self.assertIn('memory_management_tips', config)
        self.assertIn('runtime_optimization_tips', config)

    def test_monitor_training(self):
        """Test training monitoring"""
        is_safe = self.optimizer.monitor_training(
            power_limit=1000.0,  # High limit for test
            temperature_limit=100.0  # High limit for test
        )
        self.assertIsInstance(is_safe, bool)

    def test_apply_power_optimizations(self):
        """Test power optimizations application"""
        try:
            self.optimizer.apply_power_optimizations(self.model)
            optimization_applied = True
        except:
            optimization_applied = False
        self.assertTrue(optimization_applied)

    @classmethod
    def tearDownClass(cls):
        del cls.optimizer
        torch.cuda.empty_cache()

if __name__ == '__main__':
    unittest.main()
