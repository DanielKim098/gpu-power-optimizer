import tensorflow as tf
from gpu_optimizer import GPUPowerOptimizer

def create_sample_model():
    """Create a simple CNN model for demonstration"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def main():
    # Enable mixed precision for better efficiency
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Initialize our power optimizer
    power_optimizer = GPUPowerOptimizer()
    
    # Create and compile model
    model = create_sample_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Create sample data
    batch_size = 32
    sample_input = tf.random.normal((batch_size, 224, 224, 3))
    
    # Get optimal batch size
    optimal_batch_size = power_optimizer.optimize_batch_size(
        model=model,
        sample_input=sample_input,
        target_power=200.0
    )
    
    print(f"Optimal batch size for power efficiency: {optimal_batch_size}")
    
    # Get power efficiency recommendations
    config = power_optimizer.suggest_power_efficient_config()
    print("\nPower efficiency recommendations:")
    for key, value in config.items():
        print(f"\n{key}:")
        if isinstance(value, list):
            for item in value:
                print(f"- {item}")
        else:
            print(f"- {value}")
    
    # Simulate training with power monitoring
    print("\nSimulating training loop...")
    for i in range(5):
        is_safe = power_optimizer.monitor_training(
            power_limit=250.0,
            temperature_limit=80.0
        )
        
        if not is_safe:
            print("Warning: Power or temperature limits exceeded!")
            break
        
        print(f"Iteration {i+1}: Power and temperature within safe limits")
        
        # Simulate training step
        model.fit(
            sample_input,
            tf.random.uniform((batch_size,), maxval=10, dtype=tf.int32),
            epochs=1,
            verbose=0
        )

if __name__ == "__main__":
    main()
