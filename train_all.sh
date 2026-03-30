#!/bin/bash
# Train all models sequentially with roughly equal training budgets (~40 min each).
# Epoch counts tuned to wall-clock time per model:
#   U-Net:       ~4 min/epoch × 10 = ~40 min
#   SimVP:       ~4 min/epoch × 10 = ~40 min
#   ConvGRU:     ~10 min/epoch × 4 = ~40 min
#   DS-ConvLSTM: ~10 min/epoch × 4 = ~40 min
#   Flow:        ~6.5 min/epoch × 6 = ~39 min
# Total: ~3.3 hours

set -e
cd ~/projects/BOM
source venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "=== Starting training run: $(date) ==="

echo ""
echo "=== [1/5] U-Net (10 epochs, batch=16) ==="
python3 train.py --model unet --epochs 10 --batch_size 16 --lr 1e-3

echo ""
echo "=== [2/5] SimVP (10 epochs, batch=16) ==="
python3 train.py --model simvp --epochs 10 --batch_size 16 --lr 1e-3

echo ""
echo "=== [3/5] ConvGRU (4 epochs, batch=16) ==="
python3 train.py --model convgru --epochs 4 --batch_size 16 --lr 1e-3

echo ""
echo "=== [4/5] DS-ConvLSTM (4 epochs, batch=16) ==="
python3 train.py --model ds_convlstm --epochs 4 --batch_size 16 --lr 1e-3

echo ""
echo "=== [5/5] Flow (6 epochs, batch=8, lr=2e-3) ==="
python3 train.py --model flow --epochs 6 --batch_size 8 --lr 2e-3

echo ""
echo "=== All training complete: $(date) ==="

# Generate comparison visualisations for all models
echo ""
echo "=== Generating visualisations ==="
for model in unet simvp convgru ds_convlstm flow; do
    echo "  Visualising $model..."
    python3 visualise.py --model $model --n 4
done

echo "=== Done! ==="
