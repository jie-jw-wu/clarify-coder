#!/bin/bash
#SBATCH --time=8:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=107000M
#SBATCH --account=def-fard

# run this script (sbatch submit_job**.sh) in fine-tuning folder
module load python rust cuda arrow/17.0.0
source ~/ENV/bin/activate
cd $SLURM_SUBMIT_DIR
echo "we are in dir $SLURM_SUBMIT_DIR"
python clarify_aware_fine_tuning_v2.py --dataset_path /project/def-fard/jie/finetuning_data/FINAL_finetuning_data_ques_only.json --model_name_or_path /project/def-fard/jie/codellama/CodeLlama-13b-Instruct-hf   --output_dir output-dir --tokenize_version 4 --finetuned_model_path /scratch/jie/finetuned_models/CodeLlama-13b-Instruct-hf-finetuned-03132025 > result_03132025.txt