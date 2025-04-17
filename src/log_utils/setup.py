import os
import sys
import logging
import uuid
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

class TruncateFilter(logging.Filter):
    """
    Custom logging filter to truncate extremely long messages
    """
    def filter(self, record):
        if isinstance(record.msg, str) and len(record.msg) > 10000:
            record.msg = record.msg[:10000] + "... [truncated]"
        return True

def setup_logging(
    log_dir,
    log_level: str = 'INFO', 
    module: str = 'main',
    run_id: Optional[str] = None
) -> logging.Logger:
    """
    Configure advanced logging with process and run ID tracking
    
    Args:
        log_dir (str or Path): Directory to store log files
        log_level (str): Logging level
        run_id (str, optional): Unique identifier for this run
    
    Returns:
        logging.Logger: Configured logger instance
    """

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp and run ID
    log_filename = os.path.join(
        log_dir, 
        f'invoice_matcher_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )

    # formatter
    formatter = logging.Formatter(
        '%(process)d-%(thread)d [%(asctime)s.%(msecs)03d] [run:' + run_id + '] [%(module)s/%(funcName)s:%(lineno)d] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(module)
    logger.setLevel(log_level)

    # console Stream Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # file Stream Handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8', mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add custom filter
    logger.addFilter(TruncateFilter())
    
    # Redirect stdout/stderr to UTF-8
    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8')
        logger.info(f"Stdout/stderr encoding set to UTF-8")
    except Exception as e:
        logger.warning(f"Could not reconfigure stdout/stderr encoding: {e}")
    
    return logger

# Optional utility functions can be added here
def get_bangkok_time() -> datetime:
    """
    Get current time in Bangkok timezone.
    
    :return: Current datetime in Bangkok timezone
    """
    return datetime.now(ZoneInfo('Asia/Bangkok'))

# Add singleton logger class
class LoggerSingleton:
    _instance = None
    _initialized = False
    _run_id = None  # Store run ID as class variable
    
    @classmethod
    def get_logger(cls, 
                  log_dir: str = './data/logs',
                  log_level: str = 'DEBUG',
                  module: str = 'main',
                  run_id: Optional[str] = None) -> logging.Logger:
        if not cls._initialized:
            # Generate run ID if not provided and not already set
            if run_id is not None:
                cls._run_id = run_id
            elif cls._run_id is None:
                cls._run_id = str(uuid.uuid4())[:8]
                
            # Initialize logger with consistent run ID
            cls._instance = setup_logging(log_dir, log_level, module, cls._run_id)
            cls._initialized = True
        return cls._instance

def get_logger(**kwargs):
    return LoggerSingleton.get_logger(**kwargs)