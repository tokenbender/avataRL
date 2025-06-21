"""
Simple test runner to verify the generic training system works.
"""
import subprocess
import sys
import os

def run_test():
    """Run the test script"""
    print("Running avataRL generic training tests...")
    
    env = os.environ.copy()
    env['TRAINING_MODE'] = 'generic'
    env['USE_WANDB'] = 'false'
    
    try:
        result = subprocess.run([
            sys.executable, 'test_generic_training.py'
        ], env=env, cwd=os.path.dirname(__file__), capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Tests failed!")
            
        return result.returncode
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_test())
