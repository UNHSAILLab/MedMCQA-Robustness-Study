#!/bin/bash
# Setup script for Medical MCQ Robustness Study
# Run this on a new machine to set up the environment

set -e

echo "=========================================="
echo "Medical MCQ Robustness Study - Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Create virtual environment (optional)
if [ "$1" == "--venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify CUDA
echo ""
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Verify model access
echo ""
echo "Checking MedGemma model access..."
python3 -c "
from transformers import AutoTokenizer
try:
    tok = AutoTokenizer.from_pretrained('google/medgemma-4b-it')
    print('MedGemma 4B: OK')
except Exception as e:
    print(f'MedGemma 4B: FAILED - {e}')
    print('You may need to accept the model license at https://huggingface.co/google/medgemma-4b-it')
    print('And login with: huggingface-cli login')
"

# Run quick test
echo ""
echo "Running quick test..."
python3 main.py --test

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p outputs/{cache,results,figures,checkpoints}

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To run experiments:"
echo "  # Single experiment"
echo "  python main.py -e prompt_ablation -m 4b"
echo ""
echo "  # All experiments in parallel (8 GPUs)"
echo "  python scripts/run_parallel.py --gpus 8"
echo ""
echo "  # Test with limited data first"
echo "  python scripts/run_parallel.py --gpus 8 --limit 100"
echo ""
