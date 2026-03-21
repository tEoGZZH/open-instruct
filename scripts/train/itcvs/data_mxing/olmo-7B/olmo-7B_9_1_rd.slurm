#!/usr/bin/zsh
#SBATCH -J olmo7B_mixing_9_1
#SBATCH -o /home/zs473554/projects/open-instruct/scripts/train/itcvs/log/olmo7B_mixing_9_1_%j.txt
#SBATCH -t 06:00:00
#SBATCH -c 16
#SBATCH -p c23g
#SBATCH --gres=gpu:1
#SBATCH -A rwth1809

set -euo pipefail

cd /home/zs473554/projects/open-instruct
source /home/zs473554/projects/open-instruct/.venv/bin/activate
export XDG_CACHE_HOME=/work/zs473554/.cache
source /home/zs473554/jobscripts/hf_token.env

export HF_HOME=/hpcwork/zs473554/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=60

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
  bash -lc '
  export WANDB_PROJECT=OpenInstruct-OLMo-7B
  export WANDB_ENTITY=zhaohui-zeng-cn-rwth-aachen-university
  export WANDB_MODE=offline
  export WANDB_NAME="${WANDB_PROJECT}_$(date +%Y%m%d%H%M%S)"

  accelerate launch \
  --mixed_precision bf16 \
  --num_processes 1 \
  --num_machines 1 \
  --dynamo_backend no \
  open_instruct/finetune.py \
  --exp_name olmo7B_9_1_rd \
  --model_name_or_path allenai/Olmo-3-1025-7B \
  --use_slow_tokenizer \
  --add_bos True \
  --dataset_mixer_list \
  /home/zs473554/projects/InstructionTunningWithCVS/data/instruction_tuning_dataset/moral_tuning_dataset_sampled.jsonl 0.9 \
  allenai/tulu-3-sft-mixture 0.1 \
  --dataset_mixer_list_splits train \
  --output_dir /hpcwork/zs473554/checkpoints/open-instruct/olmo7B/mixing/nine_one/ \
  --do_not_randomize_output_dir \
  --add_seed_and_date_to_exp_name False \
  --max_seq_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-6 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --max_train_steps 10000 \
  --checkpointing_steps 1000 \
  --eval_steps 1000 \
  --eval_ratio 0.001 \
  --keep_last_n_checkpoints 1 \
  --logging_steps 10 \
  --seed 8 \
  --with_tracking True \
  --report_to wandb \
  --wandb_project_name OpenInstruct-OLMo-7B \
  --push_to_hub False \
  --try_launch_beaker_eval_jobs False \
  --try_auto_save_to_beaker False \
  --dataset_cache_mode local \
  --dataset_local_cache_dir /work/zs473554/open-instruct-cache'
  

