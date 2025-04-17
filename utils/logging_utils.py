"""
Standardized logging and error handling utilities for the Building Damage Pipeline.

This module provides consistent logging setup, error handling, and debugging
across all scripts in the pipeline. It implements:

1. Standard logging configuration with file and console output
2. Error handling with optional retry logic
3. Debug output formatting
4. Performance measurement tools

Usage:
    from utils.logging_utils import setup_logging, log_error, timer

    # Setup logging
    logger = setup_logging("script_name", debug=args.debug)
    
    # Log with different levels
    logger.info("Processing started")
    logger.debug("Debug details")
    
    # Time operations
    with timer("Processing video", logger):
        process_video()
        
    # Handle errors with retry
    try:
        result = handle_error_with_retry(
            function=api_call, 
            args=(param1, param2), 
            max_retries=3
        )
    except Exception as e:
        log_error(e, logger, "Failed after retries")
"""

import logging
import sys
import time
import traceback
from pathlib import Path
from functools import wraps
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import LOGS_DIR, DEBUG, ERRORS


def setup_logging(
    script_name: str, 
    debug: bool = False, 
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Set up standardized logging for a script.
    
    Args:
        script_name: Name of the script (used for logger naming)
        debug: Whether to enable debug mode logging
        log_file: Custom log file path (if None, uses script_name.log in LOGS_DIR)
        
    Returns:
        Configured logger instance
    """
    # Determine log level
    log_level = logging.DEBUG if debug or DEBUG["ENABLED"] else logging.INFO
    
    # Setup log file path
    if log_file is None:
        log_file = LOGS_DIR / f"{script_name.replace('.py', '')}.log"
    
    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(log_level)
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Log debug mode
    if debug or DEBUG["ENABLED"]:
        logger.debug(f"Debug mode enabled for {script_name}")
    
    return logger


def log_error(
    error: Exception, 
    logger: logging.Logger, 
    message: str = "An error occurred",
    include_trace: bool = None
) -> None:
    """
    Log an error with optional traceback.
    
    Args:
        error: The exception that occurred
        logger: Logger instance
        message: Custom error message
        include_trace: Whether to include full traceback (if None, uses config)
    """
    if include_trace is None:
        include_trace = ERRORS["LOG_EXCEPTIONS"]
    
    logger.error(f"{message}: {str(error)}")
    
    if include_trace:
        logger.error(traceback.format_exc())


def handle_error_with_retry(
    function: Callable, 
    args: Tuple = (), 
    kwargs: Dict = None,
    max_retries: Optional[int] = None, 
    retry_delay: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    error_message: str = "Operation failed"
) -> Any:
    """
    Execute a function with retry logic for error handling.
    
    Args:
        function: The function to call
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        max_retries: Maximum number of retry attempts (if None, uses config)
        retry_delay: Seconds between retries (if None, uses config)
        logger: Logger instance for reporting
        error_message: Custom error message
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: If all retries fail
    """
    if kwargs is None:
        kwargs = {}
    
    if max_retries is None:
        max_retries = ERRORS["MAX_RETRIES"]
        
    if retry_delay is None:
        retry_delay = ERRORS["RETRY_DELAY"]
    
    # Create minimal logger if none provided
    if logger is None:
        logger = logging.getLogger('error_handler')
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)
    
    attempt = 0
    last_error = None
    
    while attempt <= max_retries:
        try:
            return function(*args, **kwargs)
        except Exception as e:
            attempt += 1
            last_error = e
            
            if attempt <= max_retries:
                logger.warning(
                    f"{error_message}: {str(e)}. Retrying ({attempt}/{max_retries}) "
                    f"in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                # Last attempt failed
                break
    
    # If we get here, all retries failed
    logger.error(f"{error_message}: All {max_retries} retries failed.")
    raise last_error


@contextmanager
def timer(
    operation_name: str, 
    logger: Optional[logging.Logger] = None,
    debug_only: bool = True
):
    """
    Context manager for timing operations.
    
    Args:
        operation_name: Name of the operation being timed
        logger: Logger instance (if None, prints to stdout)
        debug_only: Only log timing info in debug mode
    """
    if logger is None:
        logger = logging.getLogger('timer')
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)
    
    # Skip timing if not in debug mode and debug_only is True
    if debug_only and not DEBUG["ENABLED"] and logger.level > logging.DEBUG:
        yield
        return
    
    start_time = time.time()
    logger.debug(f"Starting: {operation_name}")
    
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        
        if elapsed < 1:
            timing_str = f"{elapsed * 1000:.2f} ms"
        elif elapsed < 60:
            timing_str = f"{elapsed:.2f} seconds"
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            timing_str = f"{minutes} minutes {seconds:.2f} seconds"
        
        logger.debug(f"Completed: {operation_name} in {timing_str}")


def debug_info(
    data: Dict[str, Any], 
    logger: logging.Logger, 
    title: str = "Debug Information"
) -> None:
    """
    Log detailed debug information in a formatted way.
    
    Args:
        data: Dictionary of debug information
        logger: Logger instance
        title: Title for the debug section
    """
    if logger.level > logging.DEBUG:
        return
    
    logger.debug(f"\n{'=' * 20} {title} {'=' * 20}")
    
    for key, value in data.items():
        if isinstance(value, (list, tuple)) and len(value) > 5:
            logger.debug(f"{key}: {value[:5]} ... (total: {len(value)} items)")
        elif isinstance(value, dict) and len(value) > 5:
            logger.debug(f"{key}: {list(value.keys())[:5]} ... (total: {len(value)} keys)")
        else:
            logger.debug(f"{key}: {value}")
    
    logger.debug(f"{'=' * (42 + len(title))}")


def validate_inputs(
    inputs: Dict[str, Any], 
    requirements: Dict[str, Tuple[bool, str]], 
    logger: logging.Logger
) -> bool:
    """
    Validate inputs against requirements.
    
    Args:
        inputs: Dictionary of input values to validate
        requirements: Dictionary mapping input names to (required, error_message)
        logger: Logger instance
    
    Returns:
        True if all required inputs are valid, False otherwise
    """
    valid = True
    
    for name, (required, error_message) in requirements.items():
        # Check if required input is missing
        if required and (name not in inputs or inputs[name] is None):
            logger.error(f"Missing required input: {name}. {error_message}")
            valid = False
            continue
            
        # Skip validation for optional inputs that aren't provided
        if name not in inputs or inputs[name] is None:
            continue
        
        # Additional validation can be added here
        
    return valid 