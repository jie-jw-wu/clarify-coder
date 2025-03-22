#!/bin/bash
#SBATCH --time=6:00:00
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
python clarify_aware_fine_tuning_v2.py --dataset_path /project/def-fard/jie/finetuning_data/split_20_80_downsample.json --model_name_or_path /project/def-fard/jie/deepseek-ai/deepseek-coder-6.7b-instruct  --output_dir output-dir --tokenize_version 4 --finetuned_model_path /scratch/jie/finetuned_models/deepseek-coder-6.7b-instruct-finetuned-02252025-split_20_80_downsample > result_02252025_split_20_80_downsample.txt
#python clarify_aware_fine_tuning_v2.py --dataset_path /project/def-fard/jie/finetuning_data/split_20_80_oversample.json --model_name_or_path /project/def-fard/jie/deepseek-ai/deepseek-coder-6.7b-instruct  --output_dir output-dir --tokenize_version 4 --finetuned_model_path /scratch/jie/finetuned_models/deepseek-coder-6.7b-instruct-finetuned-02252025-split_20_80_oversample > result_02252025_split_20_80_oversample.txt