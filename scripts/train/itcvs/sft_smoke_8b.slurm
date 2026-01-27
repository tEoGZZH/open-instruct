#!/usr/bin/zsh
#SBATCH -J itcvs_sft_smoke_8b
#SBATCH -o /home/zs473554/projects/itcvs/open-instructtrain/scripts/train/itcvs/log/itcvs_sft_smoke_8b_%j.txt
#SBATCH -t 02:00:00
#SBATCH -c 16
#SBATCH -p c23g
#SBATCH --gres=gpu:1
#SBATCH -A thes2082

set -euo pipefail

cd /home/zs473554/projects/open-instruct
source /home/zs473554/projects/open-instruct/.venv/bin/activate
export PYTHONPATH=$(pwd)

srun --chdir=/home/zs473554/projects/open-instruct \
  bash -lc 'accelerate launch --mixed_precision bf16 --num_processes 1 \
    open_instruct/finetune.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --tokenizer_name meta-llama/Meta-Llama-3.1-8B \
    --use_slow_tokenizer \
    --chat_template tulu \
    --train_file /home/zs473554/projects/InstructionTunningWithCVS/data/instruction_tuning_dataset/moral_tuning_dataset.jsonl \
    --output_dir checkpoints/itcvs/tulu3/sft/8b/smoke \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --max_steps 200 \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 2 \
    --seed 8 \
    --report_to none'