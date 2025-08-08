import os
import subprocess
from utils import *


class DataGenPipeline:
    
    def __init__(self, config_path="configs/data_gen.yml"):
        """
        Initialize the pipeline with configuration.
        Automatically generates all paths based on type and base directories.
        """
        self.config = load_config(config_path)
        
        # Extract commonly used values
        self.api_key = self.config['data_gen']['API_KEY']
        self.type = self.config['data_gen']['TYPE']
        self.python_venv = self.config['data_gen']['venv']
        
        # Base directories
        self.build_dir = self.config['data_gen']['build_dir']
        self.output_dir = self.config['data_gen']['output_dir']
        self.input_dir = self.config['data_gen']['input_dir']
        
        # Type-specific directories
        self.type_upper = self.type.upper()
        self.build_type_dir = os.path.join(self.build_dir, self.type_upper)
        self.output_type_dir = os.path.join(self.output_dir, self.type_upper)
        
        # Store filenames for use in step methods
        self.filenames = self.config['data_gen']['filenames']

    def run_pipeline(self):
        """
        Run the complete data generation pipeline.
        """
        print("Starting combined script execution...")
        print(f"Processing type: {self.type}")

        # Ensure directories exist
        os.makedirs(self.build_type_dir, exist_ok=True)
        os.makedirs(self.output_type_dir, exist_ok=True)

        print(f"Build directory: {self.build_type_dir}")
        print(f"Output directory: {self.output_type_dir}")

        # Run all steps
        self.run_step2()
        self.run_convert_jsonl()
        self.run_step3()
        self.run_step4()

        print("Combined script execution completed successfully!")

        
    def run_step2(self):
        """
        Generate modified problems from the original dataset.
        """
        step2_output = f"{self.output_type_dir}/{self.filenames['step2_output'].format(type=self.type)}"
        step2_checkpoint = f"{self.build_type_dir}/{self.filenames['step2_checkpoint']}"
        
        print("Running step2.py...")
        print(f"  Input: {self.input_dir}")
        print(f"  Output: {step2_output}")
        print(f"  Checkpoint: {step2_checkpoint}")
        
        subprocess.run([
            self.python_venv, "step2.py", 
            "--api_key", self.api_key, 
            "--dir_path", self.input_dir,
            "--jsonl_file_path", step2_output,
            "--checkpoint_file", step2_checkpoint,
            "--type", self.type
        ], check=True)
        
    def run_convert_jsonl(self):
        """
        Convert the JSONL file into a format easily processed by step3.
        """
        step2_output = f"{self.output_type_dir}/{self.filenames['step2_output'].format(type=self.type)}"
        step3_input = f"{self.build_type_dir}/train"  # Move numbered folders to build
        
        print("Converting JSONL file to text format...")
        print(f"  From: {step2_output}")
        print(f"  To: {step3_input}")
        
        process_jsonl_file(step2_output, step3_input)
        print("converted jsonl file to folders and txt files")
        
    def run_step3(self):
        """
        Generate clarifying questions for the modified problems.
        """
        step3_input = f"{self.build_type_dir}/train"  # Read from build directory
        step3_output = f"{self.output_type_dir}/{self.filenames['step3_output'].format(type=self.type)}"
        step3_checkpoint = f"{self.build_type_dir}/{self.filenames['step3_checkpoint']}"
        
        print("Running step3.py...")
        print(f"  Input: {step3_input}")
        print(f"  Output: {step3_output}")
        print(f"  Checkpoint: {step3_checkpoint}")
        
        subprocess.run([
            self.python_venv, "step3.py", 
            "--api_key", self.api_key, 
            "--dir_path", step3_input,
            "--jsonl_file_path", step3_output,
            "--checkpoint_file", step3_checkpoint,
            "--type", self.type
        ], check=True)
        
    def run_step4(self):
        """
        Generate final finetuning data file.
        """
        step3_output = f"{self.output_type_dir}/{self.filenames['step3_output'].format(type=self.type)}"
        final_output = f"{self.output_type_dir}/{self.filenames['final_output'].format(type=self.type)}"
        
        print("Running step4.py...")
        print(f"  Input: {step3_output}")
        print(f"  Output: {final_output}")
        
        subprocess.run([
            self.python_venv, "step4.py", 
            step3_output, self.type, final_output
        ], check=True)
        
    # def run_separate_jsonl_with_config(self, input_file=None, output_dir_splits=None,
    #                                  output_dir_original=None, ratio=(40, 60),
    #                                  sampling_mode="oversample"):
    #     """
    #     Run separate JSONL script with optional config override.
    #     """
    #     input_file = input_file or "FINAL_finetuning_everything.jsonl"
    #     output_dir_splits = output_dir_splits or "output_splits"
    #     output_dir_original = output_dir_original or "output_original"
    #
    #     split_and_save_jsonl(input_file, output_dir_splits, ratio, sampling_mode)
    #     generate_original_type(input_file, output_dir_original)
    #
    # def run_dpo_data_format_with_config(self, input_file=None, output_file=None, max_entries=5):
    #     """
    #     Run DPO data format with optional config override.
    #     """
    #     input_file = input_file or "output_splits/split_20_80_downsample.jsonl"
    #     output_folder = "dpo_data"
    #     output_file = output_file or os.path.join(output_folder, "dpo_finetune_data.jsonl")
    #
    #     format_dpo_data(input_file, output_file, self.api_key, max_entries)
    #
    # def convert_jsonl_to_json_with_config(self, jsonl_file=None, json_file=None):
    #     """
    #     Convert JSONL to JSON with optional config override.
    #     """
    #     jsonl_file = jsonl_file or "FINAL_finetuning_data_ques_only.jsonl"
    #     json_file = json_file or "FINAL_finetuning_data_ques_only.json"
    #
    #     convert_jsonl_to_json(jsonl_file, json_file)


def main():
    """
    Main function to run the data generation pipeline.
    """
    # Initialize the pipeline with the config file
    pipeline = DataGenPipeline("configs/data_gen.yml")
    
    # Run the complete pipeline
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()