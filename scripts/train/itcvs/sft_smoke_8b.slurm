#!/usr/bin/zsh
#SBATCH -J sft_smoke_8b
#SBATCH -o /home/zs473554/projects/open-instruct/scripts/train/itcvs/log/sft_smoke_8b_eval_%j.txt
#SBATCH -t 02:00:00
#SBATCH -c 16
#SBATCH -p c23g
#SBATCH --gres=gpu:1
#SBATCH -A thes2082

set -euo pipefail

cd /home/zs473554/projects/open-instruct
source /home/zs473554/projects/open-instruct/.venv/bin/activate

# Module load and cuda home
module purge
module load CUDA/12.4.0
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

echo "CUDA_HOME=$CUDA_HOME"
which nvcc
nvcc --version

export PYTHONPATH=$(pwd)

srun --chdir=/home/zs473554/projects/open-instruct \
  bash -lc 'accelerate launch \
  --mixed_precision bf16 \
  --num_processes 1 \
  --num_machines 1 \
  --dynamo_backend no \
  open_instruct/finetune.py \
  --exp_name itcvs_tulu3_8b_sft_smoke \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
  --use_slow_tokenizer \
  --chat_template tulu \
  --dataset_mixer_list /home/zs473554/projects/InstructionTunningWithCVS/data/instruction_tuning_dataset/moral_tuning_dataset.jsonl 1.0 \
  --dataset_mixer_list_splits train \
  --output_dir checkpoints/itcvs/tulu3/sft/8b/smoke_eval \
  --do_not_randomize_output_dir \
  --add_seed_and_date_to_exp_name False \
  --max_seq_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --max_train_steps 200 \
  --checkpointing_steps 100 \
  --keep_last_n_checkpoints 1 \
  --logging_steps 10 \
  --eval_ratio 0.001 \
  --eval_steps 50 \
  --seed 8 \
  --with_tracking False \
  --report_to none \
  --push_to_hub False \
  --try_launch_beaker_eval_jobs False \
  --try_auto_save_to_beaker False'
