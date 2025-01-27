import pynvml
import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class PowerMetrics:
    power_usage: float  # Current power consumption (Watts)
    temperature: float  # Current temperature (Celsius)
    clock_speed: int   # Current clock speed (MHz)
    memory_clock: int  # Memory clock speed (MHz)

class GPUPowerOptimizer:
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.initial_metrics = self._get_current_metrics()

    def _get_current_metrics(self) -> PowerMetrics:
        """Collect current GPU state metrics"""
        info = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
        mem_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
        
        return PowerMetrics(
            power_usage=power,
            temperature=temp,
            clock_speed=clock,
            memory_clock=mem_clock
        )

    def optimize_batch_size(self, 
                          model: torch.nn.Module,
                          sample_input: torch.Tensor,
                          target_power: float) -> int:
        """Calculate optimal batch size considering power consumption"""
        current_power = self._get_current_metrics().power_usage
        
        if current_power > target_power:
            # Adjust batch size to reduce power consumption
            return max(1, int(sample_input.shape[0] * (target_power / current_power)))
        return sample_input.shape[0]

    def suggest_power_efficient_config(self) -> Dict[str, any]:
        """Suggest power-efficient settings"""
        metrics = self._get_current_metrics()
        
        return {
            'recommended_batch_size_factor': self._calculate_batch_factor(metrics),
            'memory_management_tips': [
                "Load large tensors to GPU only when needed",
                "Delete unnecessary intermediate results",
                "Use torch.no_grad() when gradient computation is not needed"
            ],
            'runtime_optimization_tips': [
                "Perform data preprocessing on CPU",
                "Use half precision (FP16) when possible",
                "Consider integrating batch normalization layers"
            ]
        }

    def monitor_training(self, 
                        power_limit: float,
                        temperature_limit: float = 80.0) -> bool:
        """Monitor power and temperature during training"""
        metrics = self._get_current_metrics()
        
        if metrics.power_usage > power_limit:
            return False
        if metrics.temperature > temperature_limit:
            return False
        return True

    def apply_power_optimizations(self, model: torch.nn.Module) -> None:
        """Apply model optimizations for power efficiency"""
        # 1. Enable Automatic Mixed Precision
        model.half()  # Convert to FP16
        
        # 2. Memory optimization
        torch.cuda.empty_cache()
        
        # 3. Set up gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

    def _calculate_batch_factor(self, metrics: PowerMetrics) -> float:
        """Calculate power-efficient batch size adjustment factor"""
        power_factor = self.initial_metrics.power_usage / metrics.power_usage
        temp_factor = 1.0 - (metrics.temperature / 100.0)
        return min(power_factor * temp_factor, 1.0)

    def __del__(self):
        pynvml.nvmlShutdown()
