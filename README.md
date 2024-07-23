# clarify-aware-coder

## Gemini Generation Pipeline

### Usage

To generate Python code from coding problem descriptions and save the output to a JSON file, we use the following arguments:

```
--api_key: API key for Google Generative AI (required).
--dir_path: Directory containing folders with coding problems (default: APPS/train).
--json_file_path: Path to save the output JSON file (default: OG-Code_Gemini_zeroshot_APPStrain.json).
```

Example:

```
python script.py --api_key YOUR_API_KEY --dir_path path/to/your/directory --json_file_path path/to/your/output.json
```
