#!/bin/bash
#SBATCH --job-name=dpo_format
#SBATCH --time=1:00:00           # Set max runtime to 1 hour
#SBATCH --nodes=1                # Request one node
#SBATCH --gpus-per-node=a100:1   # Request one A100 GPU
#SBATCH --ntasks-per-node=1      # Single task
#SBATCH --mem=32G                # Allocate 32GB RAM
#SBATCH --cpus-per-task=4        # Request 4 CPU cores
#SBATCH --account=def-fard       # Use your professor's account

module load python cuda          # Load necessary modules

source ~/ENV/bin/activate        # Activate Python virtual environment

cd /Users/arhaankhaku/Documents/Development/Projects/clarify-aware-coder  # Navigate to your script's directory

echo "Running dpo_data_format.py..."
python dpo_data_format.py > dpo_format_output.log 2>&1  # Run script and log output

echo "Job finished!"
