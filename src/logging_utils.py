"""
Structured logging utilities for the INCA project.
"""
import logging
import json
import traceback
import sys
import os
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import traceback

# Configure root logger to prevent double logging in Jupyter notebooks
logging.getLogger().setLevel(logging.WARNING)

class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs in JSON format."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_entry = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread,
            'thread_name': record.threadName,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add any extra attributes
        if hasattr(record, 'data'):
            log_entry.update(record.data)
        
        return json.dumps(log_entry, ensure_ascii=False)

def setup_logging(log_file: Optional[Union[str, Path]] = None, 
                log_level: Union[int, str] = logging.INFO,
                console: bool = True) -> None:
    """Set up logging configuration.
    
    Args:
        log_file: Path to the log file. If None, no file logging is done.
        log_level: Logging level (e.g., logging.INFO, 'DEBUG', etc.)
        console: Whether to log to console
    """
    # Convert string log level to int if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = JSONFormatter()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger with the given name.
    
    If no name is provided, the name of the calling module is used.
    
    Args:
        name: Logger name. If None, the calling module's name is used.
        
    Returns:
        Configured logger instance
    """
    if name is None:
        # Get the name of the calling module
        frame = inspect.currentframe()
        try:
            # Go up one frame to get the caller's frame
            frame = frame.f_back if frame is not None else None
            module = inspect.getmodule(frame)
            name = module.__name__ if module is not None else __name__
        finally:
            # Avoid reference cycles
            del frame
    
    return logging.getLogger(name)

class LogContext:
    """Context manager for adding context to log messages."""
    
    def __init__(self, logger: logging.Logger, **context):
        """Initialize with logger and context data.
        
        Args:
            logger: Logger instance
            **context: Context data to add to log records
        """
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        """Set up the context."""
        # Save the old record factory
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            # Create the record
            record = self.old_factory(*args, **kwargs)
            
            # Add context data
            if not hasattr(record, 'data'):
                record.data = {}
            record.data.update(self.context)
            
            return record
        
        # Set the new record factory
        logging.setLogRecordFactory(record_factory)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore the old record factory."""
        if self.old_factory is not None:
            logging.setLogRecordFactory(self.old_factory)

def log_execution_time(logger: logging.Logger, message: str = "Execution time"):
    """Decorator to log the execution time of a function.
    
    Args:
        logger: Logger instance
        message: Log message prefix
        
    Returns:
        Decorator function
    """
    def decorator(func):
        import time
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"{message}: {duration:.4f} seconds", 
                          extra={'duration_seconds': duration})
        
        return wrapper
    
    return decorator

def log_exceptions(logger: logging.Logger, reraise: bool = True):
    """Decorator to log exceptions with full traceback.
    
    Args:
        logger: Logger instance
        reraise: Whether to re-raise the exception after logging
        
    Returns:
        Decorator function
    """
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}: {str(e)}",
                    exc_info=True,
                    extra={
                        'exception_type': type(e).__name__,
                        'exception_args': str(e.args),
                        'traceback': traceback.format_exc()
                    }
                )
                if reraise:
                    raise
        
        return wrapper
    
    return decorator

def log_call(logger: logging.Logger, level: int = logging.DEBUG):
    """Decorator to log function calls with arguments and return values.
    
    Args:
        logger: Logger instance
        level: Logging level to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Format arguments
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            
            logger.log(level, f"Calling {func.__name__}({signature})")
            
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func.__name__} returned: {result!r}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {str(e)}")
                raise
        
        return wrapper
    
    return decorator
