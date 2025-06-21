"""
Modal-specific infrastructure for avataRL training.
Contains all Modal decorators, volumes, and deployment logic.
"""
import os
from pathlib import Path

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    modal = None

from config import TrainingConfig, EnvironmentConfig


def create_modal_app(config: TrainingConfig) -> 'modal.App':
    """Create Modal app with proper configuration"""
    if not MODAL_AVAILABLE:
        raise ImportError("Modal is not available. Install with: pip install modal")
    
    app = modal.App("avatarl-generic-training")
    return app


def create_modal_image() -> 'modal.Image':
    """Create Modal image with all dependencies"""
    if not MODAL_AVAILABLE:
        raise ImportError("Modal is not available")
    
    flash_attn_wheel = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("numpy", "torch==2.5.0", "tqdm", "wandb", "requests", "matplotlib", "nvidia-ml-py3")
        .pip_install(flash_attn_wheel)
    )
    return image


def create_modal_volumes() -> dict:
    """Create Modal volumes for data and checkpoints"""
    if not MODAL_AVAILABLE:
        raise ImportError("Modal is not available")
    
    volumes = {
        "/data": modal.Volume.from_name("nanogpt-data", create_if_missing=False),
        "/grpo": modal.Volume.from_name("grpo-data", create_if_missing=True)
    }
    return volumes


def create_modal_secrets() -> list:
    """Create Modal secrets list"""
    if not MODAL_AVAILABLE:
        return []
    
    return [modal.Secret.from_name("wandb-secret")]


class ModalTrainingRunner:
    """Modal-specific training runner"""
    
    def __init__(self, config: TrainingConfig, env_config: EnvironmentConfig):
        if not MODAL_AVAILABLE:
            raise ImportError("Modal is not available for Modal training mode")
        
        self.config = config
        self.env_config = env_config
        self.app = create_modal_app(config)
        self.image = create_modal_image()
        self.volumes = create_modal_volumes()
        self.secrets = create_modal_secrets()
    
    def create_training_function(self, training_func):
        """Wrap training function with Modal decorator"""
        @self.app.function(
            gpu=f"{self.config.gpu_type}:{self.config.n_gpus}",
            volumes=self.volumes,
            timeout=60 * 60 * 6,
            image=self.image,
            secrets=self.secrets,
        )
        def modal_training_wrapper():
            return training_func()
        
        return modal_training_wrapper
    
    def create_distributed_function(self, training_func):
        """Create Modal function for distributed training"""
        @self.app.function(
            gpu=f"{self.config.gpu_type}:{self.config.n_gpus}",
            volumes=self.volumes,
            timeout=60 * 60 * 6,
            image=self.image,
            secrets=self.secrets,
        )
        def modal_distributed_training():
            """Launch distributed training with torchrun"""
            import subprocess
            print(f"Launching distributed training on {self.config.n_gpus} {self.config.gpu_type} GPUs")
            
            from train_generic import ensure_dataset
            ensure_dataset(self.env_config.data_path)
            
            script_content = Path(__file__).parent.joinpath("train_generic.py").read_text()
            temp_script = "/tmp/train_multi_gpu.py"
            Path(temp_script).write_text(script_content)
            
            env = os.environ.copy()
            env['TRAINING_MODE'] = 'modal'
            
            result = subprocess.run([
                "torchrun", f"--nproc-per-node={self.config.n_gpus}", temp_script
            ], env=env)
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, result.args)
            
            return "Distributed training completed successfully!"
        
        return modal_distributed_training
    
    def run_local(self, func):
        """Run function locally with Modal"""
        return func.local()
    
    def run_remote(self, func):
        """Run function remotely on Modal"""
        return func.remote()


def setup_modal_training(config: TrainingConfig, env_config: EnvironmentConfig):
    """Setup Modal training environment"""
    if not MODAL_AVAILABLE:
        raise ImportError("Modal is not available. Please install with: pip install modal")
    
    runner = ModalTrainingRunner(config, env_config)
    return runner
