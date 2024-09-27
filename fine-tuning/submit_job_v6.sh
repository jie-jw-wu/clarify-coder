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
python clarify_aware_fine_tuning_v2.py --bs 8 --output_dir code-qwen-v5 --dataset_path /project/def-fard/jie/finetuning_data/finetuningv1.json --model_name_or_path /project/def-fard/jie/Qwen/CodeQwen1.5-7B-Chat --finetuned_model_path /project/def-fard/jie/finetuned_models/CodeQwen1.5-7B-Chat-finetuned-v5 > result_code_qwen_v5.txt
python clarify_aware_fine_tuning_v2.py --bs 8 --output_dir deepseek-chat-v5 --dataset_path /project/def-fard/jie/finetuning_data/finetuningv1.json --model_name_or_path /project/def-fard/jie/deepseek-ai/deepseek-llm-7b-chat --finetuned_model_path /project/def-fard/jie/finetuned_models/deepseek-llm-7b-chat-finetuned-v5 > result_deepseek_chat_v5.txt