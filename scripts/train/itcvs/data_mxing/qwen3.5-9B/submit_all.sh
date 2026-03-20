#!/bin/bash

# Submit all SLURM scripts in order
echo "Submitting all SLURM jobs..."
echo ""

for slurm_file in qwen3.5-9B_0_10_rd.slurm qwen3.5-9B_1_9_rd.slurm qwen3.5-9B_2_8_rd.slurm qwen3.5-9B_3_7_rd.slurm qwen3.5-9B_4_6_rd.slurm qwen3.5-9B_5_5_rd.slurm qwen3.5-9B_6_4_rd.slurm qwen3.5-9B_7_3_rd.slurm qwen3.5-9B_8_2_rd.slurm qwen3.5-9B_9_1_rd.slurm qwen3.5-9B_10_0_rd.slurm; do
    if [ -f "$slurm_file" ]; then
        echo "Submitting: $slurm_file"
        sbatch "$slurm_file"
        echo ""
    else
        echo "Warning: $slurm_file not found"
    fi
done

echo "All jobs submitted!"
echo ""
echo "Check job status with: squeue -u \$USER"
