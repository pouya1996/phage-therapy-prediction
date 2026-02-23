"""
Logging utilities with custom formatting and colors.

This module provides a centralized logging setup with:
- Standard datetime format
- Color-coded log levels (gold warnings, navy debug, red errors, bold red critical, green info)
- File and console output support
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# ANSI color codes for terminal output
class LogColors:
    """ANSI color codes for different log levels."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Log level colors
    DEBUG = "\033[34m"      # Navy blue
    INFO = "\033[32m"       # Green
    WARNING = "\033[33m"    # Gold/Yellow
    ERROR = "\033[31m"      # Red
    CRITICAL = "\033[1;31m" # Bold Red


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log messages based on level.
    """
    
    # Format string with timestamp, level, logger name, and message
    BASE_FORMAT = "%(asctime)s - %(levelname)-8s - %(name)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Color mapping for different log levels
    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }
    
    def __init__(self, use_colors: bool = True):
        """
        Initialize the formatter.
        
        Args:
            use_colors: Whether to use colors in output (disable for file logging)
        """
        super().__init__(fmt=self.BASE_FORMAT, datefmt=self.DATE_FORMAT)
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with appropriate colors.
        
        Args:
            record: The log record to format
            
        Returns:
            Formatted log message string
        """
        if self.use_colors and record.levelno in self.LEVEL_COLORS:
            # Save original levelname
            original_levelname = record.levelname
            
            # Add color to levelname
            color = self.LEVEL_COLORS[record.levelno]
            record.levelname = f"{color}{record.levelname}{LogColors.RESET}"
            
            # Format the message
            formatted = super().format(record)
            
            # Restore original levelname
            record.levelname = original_levelname
            
            return formatted
        else:
            return super().format(record)


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = False
) -> logging.Logger:
    """
    Set up a logger with colored console output and optional file output.
    
    Args:
        name: Name of the logger (typically __name__)
        log_file: Path to log file (required if file_output=True)
        level: Logging level (caller selects; default INFO)
        console_output: Whether to output to console (default: True)
        file_output: Whether to output to file (default: False)
    
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger(__name__, level=logging.DEBUG)
        >>> logger.info("This is an info message")
        >>> logger.warning("This is a warning")
        >>> logger.error("This is an error")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter(use_colors=True))
        logger.addHandler(console_handler)
    
    # File handler without colors
    if file_output:
        if log_file is None:
            raise ValueError("log_file must be provided when file_output=True")
        
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(ColoredFormatter(use_colors=False))
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger
