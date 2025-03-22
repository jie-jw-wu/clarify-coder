#!/bin/bash
#SBATCH --time=4:00:00
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
python in_dist_eval2.py --model_name_or_path /project/def-fard/jie/deepseek-ai/deepseek-coder-6.7b-instruct  --finetuned_model /project/def-fard/jie/finetuned_models/deepseek-coder-6.7b-instruct-finetuned-02212025 --dataset_path /project/def-fard/jie/finetuning_data/FINAL_finetuning_data_ques_only.json  --tokenize_version 4
