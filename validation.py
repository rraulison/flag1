"""
Input validation and data integrity checks for the INCA project.
"""
import logging
from typing import Any, Dict, List, Optional, Union, Tuple, TypeVar, Type
import pandas as pd
import numpy as np
from pathlib import Path

# Type variable for generic type hints
T = TypeVar('T')

class ValidationError(ValueError):
    """Custom exception for validation errors with structured error information."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        context_str = ''
        if self.context:
            context_str = '\nContext: ' + '; '.join(f"{k}={v}" for k, v in self.context.items())
        return f"{self.message}{context_str}"

def validate_not_none(value: Any, name: str, context: Optional[Dict[str, Any]] = None) -> None:
    """Validate that a value is not None.
    
    Args:
        value: The value to validate
        name: Name of the parameter for error messages
        context: Additional context for error messages
        
    Raises:
        ValidationError: If value is None
    """
    if value is None:
        raise ValidationError(f"{name} cannot be None", context)

def validate_type(value: Any, expected_type: Union[Type, Tuple[Type, ...]], 
                 name: str, context: Optional[Dict[str, Any]] = None) -> None:
    """Validate that a value is of the expected type.
    
    Args:
        value: The value to validate
        expected_type: Expected type or tuple of types
        name: Name of the parameter for error messages
        context: Additional context for error messages
        
    Raises:
        ValidationError: If value is not of the expected type
    """
    if not isinstance(value, expected_type):
        type_names = [t.__name__ for t in (expected_type if isinstance(expected_type, tuple) else (expected_type,))]
        raise ValidationError(
            f"{name} must be of type {' or '.join(type_names)}, got {type(value).__name__}",
            context
        )

def validate_dataframe(df: pd.DataFrame, 
                      required_columns: Optional[List[str]] = None,
                      allow_empty: bool = False,
                      name: str = "DataFrame") -> None:
    """Validate a pandas DataFrame.
    
    Args:
        df: The DataFrame to validate
        required_columns: List of columns that must be present
        allow_empty: If False, raises an error if the DataFrame is empty
        name: Name of the DataFrame for error messages
        
    Raises:
        ValidationError: If any validation fails
    """
    context = {'dataframe_name': name}
    
    # Check if DataFrame
    validate_type(df, pd.DataFrame, name, context)
    
    # Check if empty
    if not allow_empty and df.empty:
        raise ValidationError(f"{name} cannot be empty", context)
    
    # Check required columns
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            context['missing_columns'] = missing
            raise ValidationError(
                f"{name} is missing required columns: {', '.join(missing)}",
                context
            )

def validate_file_exists(file_path: Union[str, Path], 
                       name: str = "File") -> Path:
    """Validate that a file exists and return its Path object.
    
    Args:
        file_path: Path to the file
        name: Name of the file for error messages
        
    Returns:
        Path: The validated file path
        
    Raises:
        ValidationError: If the file doesn't exist or is not a file
    """
    path = Path(file_path)
    context = {'file_path': str(path.absolute())}
    
    if not path.exists():
        raise ValidationError(f"{name} does not exist: {path}", context)
    if not path.is_file():
        raise ValidationError(f"{name} is not a file: {path}", context)
    
    return path

def validate_directory_exists(dir_path: Union[str, Path], 
                            create: bool = False,
                            name: str = "Directory") -> Path:
    """Validate that a directory exists and return its Path object.
    
    Args:
        dir_path: Path to the directory
        create: If True, creates the directory if it doesn't exist
        name: Name of the directory for error messages
        
    Returns:
        Path: The validated directory path
        
    Raises:
        ValidationError: If the directory doesn't exist and create=False,
                        or if path exists but is not a directory
    """
    path = Path(dir_path)
    context = {'directory_path': str(path.absolute())}
    
    if path.exists():
        if not path.is_dir():
            raise ValidationError(f"{name} exists but is not a directory: {path}", context)
    elif create:
        path.mkdir(parents=True, exist_ok=True)
    else:
        raise ValidationError(f"{name} does not exist: {path}", context)
    
    return path

def validate_in_range(value: Union[int, float], 
                    min_val: Optional[Union[int, float]] = None, 
                    max_val: Optional[Union[int, float]] = None,
                    name: str = "Value") -> None:
    """Validate that a numeric value is within a specified range.
    
    Args:
        value: The value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the value for error messages
        
    Raises:
        ValidationError: If value is outside the specified range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
    
    if min_val is not None and value < min_val:
        raise ValidationError(
            f"{name} must be >= {min_val}, got {value}",
            {'value': value, 'min': min_val, 'max': max_val}
        )
    
    if max_val is not None and value > max_val:
        raise ValidationError(
            f"{name} must be <= {max_val}, got {value}",
            {'value': value, 'min': min_val, 'max': max_val}
        )

def validate_one_of(value: T, 
                  allowed_values: List[T], 
                  name: str = "Value") -> None:
    """Validate that a value is one of the allowed values.
    
    Args:
        value: The value to validate
        allowed_values: List of allowed values
        name: Name of the value for error messages
        
    Raises:
        ValidationError: If value is not in allowed_values
    """
    if value not in allowed_values:
        raise ValidationError(
            f"{name} must be one of {allowed_values}, got {value}",
            {'value': value, 'allowed_values': allowed_values}
        )

def validate_positive(value: Union[int, float], 
                    name: str = "Value") -> None:
    """Validate that a numeric value is positive.
    
    Args:
        value: The value to validate
        name: Name of the value for error messages
        
    Raises:
        ValidationError: If value is not positive
    """
    validate_in_range(value, min_val=0, name=name, max_val=None)
    if value == 0:
        raise ValidationError(f"{name} must be greater than 0, got {value}")

def validate_probability(value: float, 
                       name: str = "Probability") -> None:
    """Validate that a value is a valid probability (0-1).
    
    Args:
        value: The probability value to validate
        name: Name of the probability for error messages
        
    Raises:
        ValidationError: If value is not in [0, 1]
    """
    validate_in_range(value, 0.0, 1.0, name)

def validate_config(config: Any, required_attrs: List[str]) -> None:
    """Validate that a config object has all required attributes.
    
    Args:
        config: The config object to validate
        required_attrs: List of required attribute names
        
    Raises:
        ValidationError: If any required attributes are missing
    """
    missing = [attr for attr in required_attrs if not hasattr(config, attr)]
    if missing:
        raise ValidationError(
            f"Config is missing required attributes: {', '.join(missing)}",
            {'missing_attributes': missing}
        )
