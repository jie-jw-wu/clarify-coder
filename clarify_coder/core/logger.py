import logging
import os
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan (but we'll hide debug)
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Format the message first
        formatted_message = super().format(record)
        
        # Apply color to the entire message based on log level
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{formatted_message}{self.COLORS['RESET']}"
        
        return formatted_message


def setup_logger(name="clarify_coder", log_level=logging.INFO):

    # Create logger
    log = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger already exists
    if log.handlers:
        return log
    
    log.setLevel(log_level)
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create file handler for logging to file
    log_file = log_dir / f"clarify_coder_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    
    # Create console handler for colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Add formatters to handlers
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    log.addHandler(file_handler)
    log.addHandler(console_handler)
    
    return log


# Create the global logger instance
log = setup_logger()

