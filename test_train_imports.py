import os
os.environ['TRAINING_MODE'] = 'generic'
os.environ['USE_WANDB'] = 'false'

try:
    from train import is_modal_environment, GPT, setup_distributed, ensure_dataset, build_vocab
    print('✅ Train imports working')
    print(f'Modal environment: {is_modal_environment()}')
    print('✅ Core training functions imported successfully')
except Exception as e:
    print(f'❌ Train imports failed: {e}')
    import traceback
    traceback.print_exc()
