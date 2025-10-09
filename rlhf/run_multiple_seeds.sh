#!/bin/bash
# run_multiple_seeds.sh
# Train SFT with multiple random seeds for robustness

echo "========================================="
echo "Training SFT with Multiple Seeds"
echo "========================================="

# Train with 3 different seeds
for seed in 42 123 456; do
    echo ""
    echo "========================================="
    echo "Training with seed: $seed"
    echo "========================================="
    
    python3 train_sft_robust.py \
        --seed $seed \
        --output_dir ./sft_robust_output \
        --n_per_occupation 20 \
        --epochs 3 \
        --batch_size 4 \
        --grad_accum_steps 4 \
        --learning_rate 2e-5
    
    echo "Completed seed $seed"
    echo ""
done

echo "========================================="
echo "All training runs complete!"
echo "========================================="
echo ""
echo "Output directories:"
echo "  - sft_robust_output_seed42/"
echo "  - sft_robust_output_seed123/"
echo "  - sft_robust_output_seed456/"
echo ""
echo "Next steps:"
echo "  1. Generate completions: python3 generate_sft_robust.py --seed 42"
echo "  2. Analyze results across seeds"
echo "========================================="

