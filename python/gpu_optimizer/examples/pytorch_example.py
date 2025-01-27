import torch
import torchvision.models as models
from gpu_optimizer import GPUPowerOptimizer

def main():
    # Load a pretrained model
    model = models.resnet50(pretrained=True).cuda()
    
    # Initialize our power optimizer
    power_optimizer = GPUPowerOptimizer()
    
    # Apply power optimizations
    power_optimizer.apply_power_optimizations(model)
    
    # Create sample input
    batch_size = 32
    sample_input = torch.randn(batch_size, 3, 224, 224).cuda()
    
    # Get power-efficient batch size
    optimal_batch_size = power_optimizer.optimize_batch_size(
        model=model,
        sample_input=sample_input,
        target_power=200.0  # Target power consumption in watts
    )
    
    print(f"Recommended batch size: {optimal_batch_size}")
    
    # Get power efficiency suggestions
    config = power_optimizer.suggest_power_efficient_config()
    print("\nPower efficiency recommendations:")
    for key, value in config.items():
        print(f"\n{key}:")
        if isinstance(value, list):
            for item in value:
                print(f"- {item}")
        else:
            print(f"- {value}")
    
    # Monitor training loop
    print("\nSimulating training loop...")
    for i in range(5):
        # Check power consumption and temperature
        is_safe = power_optimizer.monitor_training(
            power_limit=250.0,    # Maximum power consumption in watts
            temperature_limit=80.0 # Maximum GPU temperature in Celsius
        )
        
        if not is_safe:
            print("Warning: Power or temperature limits exceeded!")
            break
            
        print(f"Iteration {i+1}: Power and temperature within safe limits")

if __name__ == "__main__":
    main()
