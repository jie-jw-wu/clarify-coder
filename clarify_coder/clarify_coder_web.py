# General Imports
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_interface.app import app
from core.logger import log

if __name__ == '__main__':
    log.info("Starting ClarifyCoder Web Interface...")
    log.info("Open your browser and go to: http://127.0.0.1:5000")
    log.info("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='127.0.0.1', port=5000)