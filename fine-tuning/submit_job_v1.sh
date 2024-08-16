#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=107000M
#SBATCH --account=def-fard

module load python rust cuda arrow/17.0.0
source ~/ENV/bin/activate
cd $SLURM_SUBMIT_DIR
echo "we are in dir $SLURM_SUBMIT_DIR"
python clarify_aware_fine_tuning_v2.py --checkpoint ./code-llama-fine-tuned-v1/checkpoint-220/adapter_model.safetensors --dataset_path /project/def-fard/jie/finetuning_data/finetuningv1.json --model_name_or_path /project/def-fard/jie/deepseek-ai/deepseek-coder-6.7b-instruct --finetuned_model_path /project/def-fard/jie/finetuned_models/deepseek-coder-6.7b-instruct-finetuned-v1 > result_v1.txt