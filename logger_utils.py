import logging
import os
from datetime import datetime

def create_logger(model_name: str) -> logging.Logger:
    """
    Creates a logger that logs everything to a file and only warnings/errors to the console.
    Also includes a message to notify where full logs are stored.
    
    Args:
        model_name (str): Used to name the log file.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{model_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Create a uniquely named logger
    logger = logging.getLogger(f"{model_name}_{timestamp}")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if the logger already exists
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # File handler (DEBUG and up)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler (WARNING and up)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Print console message manually
        print(f"Logging warnings/errors to console. See full log in {log_path}")

    return logger
