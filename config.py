"""
Configuration system for avataRL training.
Centralizes all hyperparameters and provides environment detection.
"""
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class TrainingConfig:
    """Core training configuration"""
    context_len: int = 32
    horizon: int = 8
    batch: int = 16384
    micro_batch: int = 512
    grad_accum: int = None  # Will be calculated
    total_iters: int = 2000
    lr: float = 3e-4
    min_lr: float = 3e-5
    beta_kl: float = 1e-3
    kl_warm: int = 50_000
    clip_ratio: float = 0.2
    k_samples: int = 4
    entropy_coef: float = 0.01
    temperature: float = 1.2
    min_variance: float = 0.1
    
    # Model architecture
    n_layer: int = 6
    n_head: int = 8
    n_emb: int = 512
    
    # Optimization settings
    use_flash_attn: bool = True
    use_torch_compile: bool = False
    use_chunked_loss: bool = False
    use_8bit_optimizer: bool = False
    
    # Advanced settings
    adaptive_kl: bool = True
    kl_target: float = 0.02
    grad_clip: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    
    use_lr_decay: bool = True
    lr_decay_type: str = "cosine"  # "cosine", "linear", "exponential"
    warmup_iters: int = 100
    
    log_interval: int = 5
    eval_interval: int = 20
    sample_interval: int = 20
    save_intermediate_checkpoints: bool = True
    save_final_checkpoint: bool = True
    checkpoint_interval: int = 100
    
    use_exhaustive: bool = True
    use_confidence_scaling: bool = False
    confidence_weight: float = 0.1
    
    n_gpus: int = 1
    gpu_type: str = "A100"
    bucket_size_mb: float = 25.0
    
    def __post_init__(self):
        if self.grad_accum is None:
            self.grad_accum = self.batch // self.micro_batch
    
    @classmethod
    def from_env(cls) -> 'TrainingConfig':
        """Create config from environment variables"""
        config = cls()
        
        env_mappings = {
            'CONTEXT_LEN': 'context_len',
            'HORIZON': 'horizon', 
            'BATCH': 'batch',
            'MICRO_BATCH': 'micro_batch',
            'TOTAL_ITERS': 'total_iters',
            'LR': 'lr',
            'N_LAYER': 'n_layer',
            'N_HEAD': 'n_head',
            'N_EMB': 'n_emb',
            'N_GPUS': 'n_gpus',
            'GPU_TYPE': 'gpu_type',
        }
        
        for env_var, attr_name in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                if hasattr(config, attr_name):
                    attr_type = type(getattr(config, attr_name))
                    if attr_type == bool:
                        setattr(config, attr_name, value.lower() in ('true', '1', 'yes'))
                    else:
                        setattr(config, attr_name, attr_type(value))
        
        config.grad_accum = config.batch // config.micro_batch
        return config


@dataclass 
class EnvironmentConfig:
    """Environment-specific configuration"""
    training_mode: str = "generic"  # "modal" or "generic"
    use_wandb: bool = True
    wandb_project: str = "avataRL"
    wandb_name: Optional[str] = None
    data_path: str = "input.txt"
    checkpoint_dir: str = "./checkpoints"
    
    @classmethod
    def from_env(cls) -> 'EnvironmentConfig':
        """Create environment config from environment variables"""
        config = cls()
        
        config.training_mode = os.environ.get('TRAINING_MODE', 'generic').lower()
        
        if config.training_mode == 'modal':
            config.checkpoint_dir = "/data"
            config.data_path = "input.txt"
        
        config.use_wandb = os.environ.get('USE_WANDB', 'true').lower() in ('true', '1', 'yes')
        config.wandb_project = os.environ.get('WANDB_PROJECT', config.wandb_project)
        config.wandb_name = os.environ.get('WANDB_NAME', config.wandb_name)
        config.data_path = os.environ.get('DATA_PATH', config.data_path)
        config.checkpoint_dir = os.environ.get('CHECKPOINT_DIR', config.checkpoint_dir)
        
        return config


def get_config() -> tuple[TrainingConfig, EnvironmentConfig]:
    """Get complete configuration from environment"""
    training_config = TrainingConfig.from_env()
    env_config = EnvironmentConfig.from_env()
    return training_config, env_config


def is_modal_environment() -> bool:
    """Check if running in Modal environment"""
    return os.environ.get('TRAINING_MODE', '').lower() == 'modal' or 'MODAL_' in os.environ
