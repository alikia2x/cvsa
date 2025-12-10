"""
Unified logging configuration for ml_new training project
Provides colorful level formatting with [level]: [msg] format
"""
import logging
import sys

class ColorfulFormatter(logging.Formatter):
    """Custom formatter with colorful level names and [level]: [msg] format"""
    
    # ANSI color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'  # Reset color
    
    def format(self, record):
        # Extract level name
        level_name = record.levelname
        
        # Apply color to level name
        color = self.COLORS.get(level_name, self.RESET)
        colored_level = f"{color}{level_name}{self.RESET}"
        
        # Create new format: [colored_level]: [msg]
        # Get the message part only (without level name, module, etc.)
        if hasattr(record, 'message'):
            message = record.message
        else:
            message = record.getMessage()
        
        # Return in the requested format
        return f"{colored_level}:     {message}"


class ColoredConsoleHandler(logging.StreamHandler):
    """Console handler that outputs to stderr with colors"""
    
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stderr
        super().__init__(stream)
        self.setFormatter(ColorfulFormatter())


def setup_logger(
    name: str = None,
    level: str = "INFO",
    log_file: str = None,
    console_output: bool = True,
    file_output: bool = False
) -> logging.Logger:
    """
    Setup a unified logger with colorful formatting
    
    Args:
        name: Logger name (defaults to project name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        console_output: Whether to output to console
        file_output: Whether to output to file
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler with colorful formatting
    if console_output:
        console_handler = ColoredConsoleHandler()
        logger.addHandler(console_handler)
    
    # File handler with standard formatting (no colors in files)
    if file_output and log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '[%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance with unified configuration
    
    Args:
        name: Logger name (defaults to 'ml_new.training')
    
    Returns:
        Logger instance
    """
    if name is None:
        name = 'ml_new.training'
    
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        return setup_logger(name=name)
    
    return logger


# Project-wide logger configuration
def configure_project_logger():
    """Configure the main project logger with all modules"""
    
    # Configure root logger for the project
    root_logger = logging.getLogger('ml_new')
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    console_handler = ColoredConsoleHandler()
    root_logger.addHandler(console_handler)
    
    # Prevent propagation to avoid duplicates
    root_logger.propagate = False
    
    return root_logger


# Convenience function for quick setup
def quick_setup(level: str = "INFO") -> logging.Logger:
    """
    Quick setup for individual modules
    
    Args:
        level: Logging level
    
    Returns:
        Logger instance for the calling module
    """
    # Get the calling module name
    import inspect
    frame = inspect.currentframe().f_back
    module_name = frame.f_globals.get('__name__', 'ml_new.training')
    
    return setup_logger(name=module_name, level=level)


if __name__ == "__main__":
    # Test the logger configuration
    logger = get_logger('test_logger')
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")