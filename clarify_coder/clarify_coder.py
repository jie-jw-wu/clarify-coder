# General Imports
import os
# Local Imports
from core import utils
from core.logger import log


class ClarifyCoder:
    
    def __init__(self, config_path: str = "./configs/clarify_coder.yml"):
        self._config = utils.load_config(config_path)
        
        # Set Variables
        self._workspace = os.getcwd()
        self._build = f'{self._workspace}/build'
        
        # Get Variables
        self.interface = self._config['clarify_coder']['interface']
        self.output_dir = self._config['clarify_coder']['output_dir']
      
        
        # Initialize LLM interface
        self.llm = utils.get_llm(self.interface)
    
    def run_pipeline(self):
        log.info("Running Clarify Coder pipeline...")
        response = self.llm.get_response("Hello, how are you?")
        log.info(f"LLM Response: {response}")

    
    
if __name__ == "__main__":
    app = ClarifyCoder()
    app.run_pipeline()