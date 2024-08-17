#!/bin/bash
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=107000M
#SBATCH --account=def-fard

module load python rust cuda arrow/17.0.0
source ~/ENV/bin/activate
cd $SLURM_SUBMIT_DIR
echo "we are in dir $SLURM_SUBMIT_DIR"
python clarify_aware_fine_tuning_v2.py --output_dir code-llama-v2 --dataset_path /project/def-fard/jie/finetuning_data/finetuningv1.json --model_name_or_path /project/def-fard/jie/codellama/CodeLlama-7b-Instruct-hf --finetuned_model_path /project/def-fard/jie/finetuned_models/CodeLlama-7b-Instruct-hf-finetuned-v2 > result_codellama_v2.txt