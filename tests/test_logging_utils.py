import unittest
import sys
import os
import logging
import time
from pathlib import Path
import tempfile
import io
from contextlib import redirect_stdout

# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.logging_utils import (
    setup_logging,
    log_error,
    handle_error_with_retry,
    timer,
    debug_info,
    validate_inputs
)
from config import *

class TestLoggingUtils(unittest.TestCase):
    """Unit tests for logging utility functions"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.log_file = self.temp_dir / "test_log.log"
    
    def tearDown(self):
        """Clean up after tests"""
        import shutil
        # Close and remove any handlers
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
        shutil.rmtree(self.temp_dir)
    
    def test_setup_logging(self):
        """Test the setup_logging function"""
        # Test with debug mode
        logger = setup_logging("test_logger", debug=True, log_file=self.log_file)
        
        # Check logger level
        self.assertEqual(logger.level, logging.DEBUG)
        
        # Check handlers
        self.assertEqual(len(logger.handlers), 2)  # File and console handlers
        
        # Write some logs
        logger.debug("Debug message")
        logger.info("Info message")
        
        # Check log file contents
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            self.assertIn("Debug message", log_content)
            self.assertIn("Info message", log_content)
    
    def test_log_error(self):
        """Test the log_error function"""
        # Setup a test logger with a string buffer
        logger = logging.getLogger("test_error_logger")
        logger.setLevel(logging.ERROR)
        
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger.addHandler(handler)
        
        # Test with a simple error
        error = ValueError("Test error")
        log_error(error, logger, "Custom error message")
        
        # Check the output
        log_output = log_stream.getvalue()
        self.assertIn("Custom error message", log_output)
        self.assertIn("Test error", log_output)
    
    def test_handle_error_with_retry(self):
        """Test the error handling with retry function"""
        # Counter for function calls
        counter = {"calls": 0, "success_on": 3}
        
        # Mock function that succeeds after a few retries
        def mock_function():
            counter["calls"] += 1
            if counter["calls"] < counter["success_on"]:
                raise ValueError(f"Failure on attempt {counter['calls']}")
            return "Success"
        
        # Setup a test logger
        logger = logging.getLogger("test_retry_logger")
        
        # Test with retries
        result = handle_error_with_retry(
            function=mock_function,
            max_retries=3,
            retry_delay=0.1,  # Small delay for testing
            logger=logger
        )
        
        # Check the result
        self.assertEqual(result, "Success")
        self.assertEqual(counter["calls"], counter["success_on"])
        
        # Test with a function that always fails
        counter = {"calls": 0}
        
        def always_fails():
            counter["calls"] += 1
            raise ValueError("Always fails")
        
        # Should raise the exception after all retries
        with self.assertRaises(ValueError):
            handle_error_with_retry(
                function=always_fails,
                max_retries=2,
                retry_delay=0.1,
                logger=logger
            )
        
        # Should have tried 3 times (initial + 2 retries)
        self.assertEqual(counter["calls"], 3)
    
    def test_timer(self):
        """Test the timer context manager"""
        # Setup a test logger
        logger = logging.getLogger("test_timer_logger")
        logger.setLevel(logging.DEBUG)
        
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger.addHandler(handler)
        
        # Use the timer
        with timer("Test operation", logger, debug_only=False):
            time.sleep(0.1)  # Small sleep for timing
        
        # Check the log output
        log_output = log_stream.getvalue()
        self.assertIn("Starting: Test operation", log_output)
        self.assertIn("Completed: Test operation", log_output)
        self.assertIn("seconds", log_output)
    
    def test_debug_info(self):
        """Test the debug_info function"""
        # Setup a test logger
        logger = logging.getLogger("test_debug_logger")
        logger.setLevel(logging.DEBUG)
        
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger.addHandler(handler)
        
        # Test with a dictionary
        test_data = {
            "key1": "value1",
            "key2": 123,
            "key3": [1, 2, 3]
        }
        
        debug_info(test_data, logger, "Test Debug Info")
        
        # Check the log output
        log_output = log_stream.getvalue()
        self.assertIn("Test Debug Info", log_output)
        self.assertIn("key1", log_output)
        self.assertIn("value1", log_output)
        self.assertIn("key2", log_output)
        self.assertIn("123", log_output)
    
    def test_validate_inputs(self):
        """Test the validate_inputs function"""
        # Setup a test logger
        logger = logging.getLogger("test_validate_logger")
        
        # Test with valid inputs
        inputs = {
            "required_string": "test",
            "required_int": 123,
            "optional_value": None
        }
        
        requirements = {
            "required_string": (True, "string"),
            "required_int": (True, "int"),
            "optional_value": (False, "any"),
            "missing_optional": (False, "string")
        }
        
        # Should return True for valid inputs
        self.assertTrue(validate_inputs(inputs, requirements, logger))
        
        # Test with missing required value
        invalid_inputs = {
            "required_string": "test",
            # Missing required_int
            "optional_value": None
        }
        
        # Should return False
        self.assertFalse(validate_inputs(invalid_inputs, requirements, logger))
        
        # Test with wrong type
        invalid_type = {
            "required_string": 123,  # Should be string, not int
            "required_int": 123,
            "optional_value": None
        }
        
        # Should return False
        self.assertFalse(validate_inputs(invalid_type, requirements, logger))

if __name__ == '__main__':
    unittest.main() 