import os
from datetime import datetime
import logging

log_filename = datetime.now().strftime("log/agent_logs_%Y-%m-%d_%H-%M-%S.log")
# Create the directory if it doesn't exist
log_dir = os.path.dirname(log_filename)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

print(f"Log filename: {log_filename}")

class HTTPFilter(logging.Filter):
    """Filter out log records containing unwanted messages."""
    def filter(self, record):
        message = record.getMessage()
        return "HTTP Request" not in message and "ucb_score" not in message

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.addFilter(HTTPFilter())
    logger.addHandler(file_handler)
