# General Imports
import google.generativeai as genai
import os

# Local Imports
from core.logger import log


class GoogleGeminiInterface:
    
    def __init__(self, api_key = None, model_name = "default"):
        self.api_key = os.getenv('GOOGLE_API_KEY') or api_key
        self.model_name = os.getenv('GOOGLE_MODEL_NAME') or model_name
        
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(self.model_name)
    
    def get_response(self, prompt):
        try:
            log.debug(f"Prompt: {prompt} getting response from llm")
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            log.error(f"Error generating response from Gemini: {str(e)}")
            raise Exception(f"Error generating response from Gemini: {str(e)}")