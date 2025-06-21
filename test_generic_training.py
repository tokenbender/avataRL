"""
Test script to verify generic training functionality.
"""
import os
import sys
import tempfile
from pathlib import Path

def test_config_system():
    """Test the configuration system"""
    print("Testing configuration system...")
    
    try:
        from config import TrainingConfig, EnvironmentConfig, get_config
        
        config, env_config = get_config()
        assert config.context_len == 32
        assert config.horizon == 8
        assert env_config.training_mode == "generic"
        
        os.environ['CONTEXT_LEN'] = '64'
        os.environ['TRAINING_MODE'] = 'modal'
        config, env_config = get_config()
        assert config.context_len == 64
        assert env_config.training_mode == 'modal'
        
        print("✓ Configuration system working")
        return True
    except Exception as e:
        print(f"✗ Configuration system failed: {e}")
        return False

def test_modal_imports():
    """Test optional Modal imports"""
    print("Testing Modal imports...")
    
    try:
        from modal_runner import MODAL_AVAILABLE
        print(f"✓ Modal available: {MODAL_AVAILABLE}")
        return True
    except Exception as e:
        print(f"✗ Modal import failed: {e}")
        return False

def test_train_imports():
    """Test that train.py imports work"""
    print("Testing train.py imports...")
    
    try:
        os.environ['TRAINING_MODE'] = 'generic'
        os.environ['USE_WANDB'] = 'false'
        
        from train import GPT, setup_distributed, ensure_dataset, build_vocab
        print("✓ Core training imports working")
        return True
    except Exception as e:
        print(f"✗ Train imports failed: {e}")
        return False

def test_generic_training_setup():
    """Test generic training setup"""
    print("Testing generic training setup...")
    
    try:
        os.environ['TRAINING_MODE'] = 'generic'
        os.environ['USE_WANDB'] = 'false'
        
        from train_generic import setup_distributed, ensure_dataset, build_vocab
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            ensure_dataset(temp_path)
            assert Path(temp_path).exists()
            
            encode, decode, vocab_size, stoi, itos, text = build_vocab(temp_path)
            assert vocab_size > 0
            assert len(text) > 0
            
            print("✓ Generic training setup working")
            return True
        finally:
            Path(temp_path).unlink(missing_ok=True)
            
    except Exception as e:
        print(f"✗ Generic training setup failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running avataRL generic training tests...\n")
    
    tests = [
        test_config_system,
        test_modal_imports,
        test_train_imports,
        test_generic_training_setup,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Generic training system is working.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
