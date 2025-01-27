import unittest
import torch
import tensorflow as tf
from gpu_optimizer import GPUPowerOptimizer

class TestFrameworkIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = GPUPowerOptimizer()
        cls.batch_size = 16
        
    def test_pytorch_integration(self):
        """Test integration with PyTorch"""
        # Setup PyTorch model and data
        try:
            import torchvision.models as models
            model = models.resnet18(pretrained=False).cuda()
            inputs = torch.randn(self.batch_size, 3, 224, 224).cuda()
            
            # Test optimizations
            self.optimizer.apply_power_optimizations(model)
            
            # Run inference
            with torch.no_grad():
                outputs = model(inputs)
            
            self.assertEqual(outputs.shape[0], self.batch_size)
            return True
        except Exception as e:
            self.fail(f"PyTorch integration failed: {str(e)}")
    
    def test_tensorflow_integration(self):
        """Test integration with TensorFlow"""
        try:
            # Enable mixed precision
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            # Create simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', 
                                     input_shape=(224, 224, 3)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(10)
            ])
            
            # Create sample data
            inputs = tf.random.normal((self.batch_size, 224, 224, 3))
            
            # Test optimizer integration
            optimal_batch = self.optimizer.optimize_batch_size(
                model=model,
                sample_input=inputs,
                target_power=200.0
            )
            
            self.assertGreater(optimal_batch, 0)
            return True
        except Exception as e:
            self.fail(f"TensorFlow integration failed: {str(e)}")
    
    def test_cross_framework_monitoring(self):
        """Test monitoring works across frameworks"""
        for _ in range(3):
            is_safe = self.optimizer.monitor_training(
                power_limit=250.0,
                temperature_limit=80.0
            )
            self.assertIsInstance(is_safe, bool)
    
    def test_gpu_memory_management(self):
        """Test GPU memory handling across frameworks"""
        initial_memory = torch.cuda.memory_allocated()
        
        # Run some operations
        torch_tensor = torch.randn(1000, 1000).cuda()
        tf_tensor = tf.random.normal((1000, 1000))
        
        # Check memory is properly managed
        self.optimizer.apply_power_optimizations(None)  # Trigger cleanup
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        self.assertLessEqual(final_memory, initial_memory * 1.1)  # Allow 10% overhead
    
    @classmethod
    def tearDownClass(cls):
        del cls.optimizer
        torch.cuda.empty_cache()
        tf.keras.backend.clear_session()

if __name__ == '__main__':
    unittest.main()
