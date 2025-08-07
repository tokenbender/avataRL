#!/bin/bash
# avataRL Server Setup Script
# This script prepares a new server for training with train.py
# Environment parity with Modal: Python 3.11, PyTorch 2.5.0, Flash Attention 2.6.3

set -e  # Exit on error

echo "avataRL Server Setup Script"
echo "=============================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print colored output
print_status() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Step 1: Check CUDA availability
echo -e "\n${GREEN}Step 1: Checking CUDA installation${NC}"
if command_exists nvidia-smi; then
    print_status "NVIDIA driver detected"
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | sed 's/^/  /'
else
    print_error "nvidia-smi not found. Please install NVIDIA drivers and CUDA."
    exit 1
fi

# Step 2: Check Python version (Modal uses Python 3.11)
echo -e "\n${GREEN}Step 2: Checking Python installation${NC}"
if command_exists python3.11; then
    PYTHON_CMD=python3.11
    print_status "Python 3.11 detected (Modal parity)"
elif command_exists python3; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -eq 11 ]; then
        PYTHON_CMD=python3
        print_status "Python 3.11 detected (Modal parity)"
    else
        print_warning "Python $PYTHON_VERSION detected. Modal uses Python 3.11"
        print_warning "Installing dependencies may result in different behavior"
        PYTHON_CMD=python3
    fi
else
    print_error "Python3 not found. Please install Python 3.11 for Modal parity"
    exit 1
fi

# Step 3: Create virtual environment
echo -e "\n${GREEN}Step 3: Setting up Python virtual environment${NC}"
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    print_status "Created virtual environment"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Activated virtual environment"

# Step 4: Upgrade pip and install packaging (Modal parity)
echo -e "\n${GREEN}Step 4: Installing base packages${NC}"
pip install --upgrade pip packaging wheel setuptools
print_status "Base packages installed"

# Step 5: Install PyTorch with CUDA support
echo -e "\n${GREEN}Step 5: Installing PyTorch with CUDA support${NC}"
# Get CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
echo "Detected CUDA Version: $CUDA_VERSION"

# Install PyTorch based on CUDA version
if [[ "$CUDA_VERSION" == "12"* ]]; then
    print_status "Installing PyTorch for CUDA 12.x"
    pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
elif [[ "$CUDA_VERSION" == "11"* ]]; then
    print_status "Installing PyTorch for CUDA 11.x"
    pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu118
else
    print_warning "Unknown CUDA version. Installing CPU-only PyTorch"
    pip install torch==2.5.0
fi

# Step 6: Install other dependencies (Modal parity - no version pinning except torch)
echo -e "\n${GREEN}Step 6: Installing project dependencies${NC}"
pip install numpy tqdm wandb requests matplotlib nvidia-ml-py3

# Step 7: Install Flash Attention
echo -e "\n${GREEN}Step 7: Installing Flash Attention${NC}"
# Modal uses precompiled wheel for Python 3.11, CUDA 12.3, PyTorch 2.3
# For local install, we'll try the wheel first, then fall back to source build
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

print_warning "Installing Flash Attention (this may take several minutes)..."
if [[ "$PYTHON_CMD" == "python3.11" ]] && [[ "$CUDA_VERSION" == "12"* ]]; then
    print_status "Attempting to use Modal's precompiled wheel"
    if pip install $FLASH_ATTN_WHEEL; then
        print_status "Flash Attention installed from wheel"
    else
        print_warning "Wheel installation failed, building from source"
        pip install flash-attn==2.6.3 --no-build-isolation
        print_status "Flash Attention built from source"
    fi
else
    print_warning "Python/CUDA version mismatch with Modal wheel, building from source"
    pip install flash-attn==2.6.3 --no-build-isolation
    print_status "Flash Attention built from source"
fi

# Step 8: Verify PyTorch CUDA
echo -e "\n${GREEN}Step 8: Verifying PyTorch CUDA support${NC}"
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    print_status "PyTorch CUDA support verified"
    print_status "Number of GPUs available: $GPU_COUNT"
else
    print_error "PyTorch CUDA support not available. Check CUDA installation."
fi

# Step 9: Download dataset if not present
echo -e "\n${GREEN}Step 9: Checking for dataset${NC}"
if [ ! -f "input.txt" ]; then
    print_warning "Dataset not found. Downloading tinyshakespeare..."
    wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    print_status "Dataset downloaded"
else
    print_status "Dataset already exists"
fi

# Step 10: Create launch scripts
echo -e "\n${GREEN}Step 10: Creating launch scripts${NC}"

# Single GPU launch script
cat > launch_single_gpu.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
echo "Starting single GPU training..."
python train.py
EOF
chmod +x launch_single_gpu.sh
print_status "Created launch_single_gpu.sh"

# Multi GPU launch script
cat > launch_multi_gpu.sh << 'EOF'
#!/bin/bash
source venv/bin/activate

# Auto-detect number of GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected $GPU_COUNT GPUs"

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "No GPUs detected. Exiting."
    exit 1
fi

echo "Starting distributed training on $GPU_COUNT GPUs..."
torchrun --nproc_per_node=$GPU_COUNT train.py
EOF
chmod +x launch_multi_gpu.sh
print_status "Created launch_multi_gpu.sh"

# Step 11: Final verification
echo -e "\n${GREEN}Setup Complete!${NC}"
echo "================"
echo -e "\nEnvironment Summary:"
python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo -e "\n${GREEN}Next Steps:${NC}"
echo "1. Activate environment: source venv/bin/activate"
echo "2. For single GPU: ./launch_single_gpu.sh"
echo "3. For multi-GPU: ./launch_multi_gpu.sh"
echo "4. Monitor with wandb (optional): Set WANDB_API_KEY environment variable"

echo -e "\n${YELLOW}Note:${NC} If you encounter issues with Flash Attention, you may need to:"
echo "  - Ensure CUDA toolkit is installed (not just drivers)"
echo "  - Install ninja: pip install ninja"
echo "  - Check GCC version (requires GCC 7+)"