import os
os.environ['TRAINING_MODE'] = 'generic'
os.environ['USE_WANDB'] = 'false'

try:
    from config import get_config
    config, env_config = get_config()
    print('✅ Configuration system working')
    print(f'Training mode: {env_config.training_mode}')
    print(f'Context length: {config.context_len}')
    print(f'Batch size: {config.batch}')
except Exception as e:
    print(f'❌ Configuration failed: {e}')
    import traceback
    traceback.print_exc()
