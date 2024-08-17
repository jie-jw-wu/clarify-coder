#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=107000M
#SBATCH --account=def-fard

module load python rust cuda arrow/17.0.0
source ~/ENV/bin/activate
cd $SLURM_SUBMIT_DIR
echo "we are in dir $SLURM_SUBMIT_DIR"
python clarify_aware_fine_tuning_v2.py --output_dir code-llama-v4 --dataset_path /project/def-fard/jie/finetuning_data/finetuningv1.json --model_name_or_path /project/def-fard/jie/codellama/CodeLlama-7b-Instruct-hf --finetuned_model_path /project/def-fard/jie/finetuned_models/CodeLlama-7b-Instruct-hf-finetuned-v4 > result_codellama_v4.txt
python clarify_aware_fine_tuning_v2.py --output_dir deepseek-coder-v4 --dataset_path /project/def-fard/jie/finetuning_data/finetuningv1.json --model_name_or_path /project/def-fard/jie/deepseek-ai/deepseek-coder-6.7b-instruct --finetuned_model_path /project/def-fard/jie/finetuned_models/deepseek-coder-6.7b-instruct-finetuned-v4 > result_deepseek_coder_v4.txt
python clarify_aware_fine_tuning_v2.py --output_dir code-qwen-v4 --dataset_path /project/def-fard/jie/finetuning_data/finetuningv1.json --model_name_or_path /project/def-fard/jie/Qwen/CodeQwen1.5-7B-Chat --finetuned_model_path /project/def-fard/jie/finetuned_models/CodeQwen1.5-7B-Chat-finetuned-v4 > result_code_qwen_v4.txt