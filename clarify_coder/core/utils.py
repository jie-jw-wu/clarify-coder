# General Imports
import yaml
from pathlib import Path

# Local Imports
from interfaces.google_gemini import GoogleGeminiInterface

def load_config(config_path):
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load the main config file
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file) or {}
    
    # Check for includes
    if 'includes' in config:
        includes = config['includes']
        if isinstance(includes, list):
            for include_path in includes:
                # Resolve relative paths relative to the main config file
                include_full_path = config_path.parent / include_path
                
                if include_full_path.exists():
                    with open(include_full_path, 'r', encoding='utf-8') as include_file:
                        include_config = yaml.safe_load(include_file) or {}
                        # Merge the included config into the main config
                        config = merge_configs(config, include_config)
                else:
                    print(f"Warning: Include file not found: {include_full_path}")
        
        # Remove the includes key from the final config
        config.pop('includes', None)
    
    return config


def merge_configs(base_config, include_config):
    merged = base_config.copy()
    
    for key, value in include_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_llm(interface_name):
    # Find interface
    if interface_name.startswith('gemini'):
        llm = GoogleGeminiInterface() 
    # Add more interfaces here..
    else:
        raise ValueError(f"Unsupported interface: {interface_name}. Supported interfaces: gemini")
    
    print(f"Pipeline initialized with {interface_name} interface")
    return llm