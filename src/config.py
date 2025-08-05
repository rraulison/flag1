# this file handles the configuration of variables - config.py

"""
Enhanced configuration with better organization and validation.
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import logging
import platform
import subprocess
import sys
from datetime import datetime

def detect_gpu() -> Dict[str, Any]:
    """Detect available GPU and return configuration."""
    gpu_info = {
        'available': False,
        'type': None,
        'count': 0,
        'devices': []
    }
    
    try:
        # Try to detect NVIDIA GPUs
        if platform.system() == 'Windows':
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                     capture_output=True, text=True, check=True)
                if result.returncode == 0:
                    gpu_info['available'] = True
                    gpu_info['type'] = 'nvidia'
                    gpu_info['devices'] = [line.split(',')[0].strip() for line in result.stdout.strip().split('\n')]
                    gpu_info['count'] = len(gpu_info['devices'])
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
        else:  # Linux/Unix
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    gpu_info['available'] = True
                    gpu_info['type'] = 'nvidia'
                    gpu_info['devices'] = [line.split(',')[0].strip() for line in result.stdout.strip().split('\n') if line.strip()]
                    gpu_info['count'] = len(gpu_info['devices'])
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
        
        # If no NVIDIA GPU, check for AMD (ROCm)
        if not gpu_info['available']:
            try:
                result = subprocess.run(['rocm-smi', '--showid'], 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    gpu_info['available'] = True
                    gpu_info['type'] = 'amd'
                    # Count the number of GPU devices
                    gpu_info['count'] = len([line for line in result.stdout.split('\n') 
                                           if 'GPU[' in line and 'GPU ID' not in line])
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
                
    except Exception as e:
        logging.warning(f"Error detecting GPU: {str(e)}")
        
    return gpu_info

class Config:
    """Centralized configuration with validation and hardware detection."""
    
    # Determine the base directory of the project dynamically
    _BASE_DIR = Path(__file__).parent.parent.resolve()
    
    # Detect hardware configuration
    _GPU_INFO = detect_gpu()
    
    # Performance settings based on available hardware
    USE_GPU: bool = _GPU_INFO['available']
    DEVICE_TYPE: str = 'gpu' if _GPU_INFO['available'] else 'cpu'
    N_JOBS: int = -1  # Use all available CPU cores by default
    
    # Temperature scaling for probability calibration
    TEMPERATURE_SCALING: float = 0.5  # Lower values = softer probabilities

    # Core Settings
    DEBUG: bool = True
    N_IMPUTATIONS: int = 100  # Reduced from 100 for testing
    CONFIDENCE_LEVEL: float = 0.95
    TARGET_VARIABLE: str = 'ESTADIAM'
    ORDINAL_TARGET_VARIABLE: str = 'estadiamento_ordinal'
    
    # Class labels for the target variable (ESTADIAM)
    # These should match the possible values in your target variable
    CLASSES: List[str] = ['0', '1', '2', '3', '4']  # Possible stages 0-4
    
    # Performance Settings
    RECIPIENT_QUERY_BATCH_SIZE: int = 50000  # Reduced batch size for better memory management
    SUBSAMPLE_RATIO: float = 0.8  # Slightly reduced for faster training
    EARLY_STOPPING_ROUNDS: int = 20  # Reduced for faster convergence
    K_PMM_NEIGHBORS: int = 10  # Reduced number of neighbors for faster PMM
    

    # Data Settings
    DATA_FILE: Path = _BASE_DIR / 'data' / 'df_preprocessed.parquet'
    FEATURE_IMPORTANCE_FILE: Path = _BASE_DIR / 'data' / 'results' / 'selected_features_cv.csv'
    OUTPUT_DIR: Path = _BASE_DIR / 'outputs'
    
    # Validation Rules
    VALID_T: List[str] = ['0', '1', '2', '3', '4', '8', '9']
    VALID_N: List[str] = ['0', '1', '2', '3', '8', '9']
    VALID_M: List[str] = ['0', '1', '8', '9']
    VALID_TUMOR: List[str] = ['0', '1', '2', '3', '6', '9']
    
    # MNAR Scenarios
    MNAR_SCENARIOS: Dict[str, Dict[str, Union[float, List[str]]]] = {
        "mnar_adv": {"delta": 1.5, "stages": ["3", "4"]},
        "mnar_s4": {"delta": 3.0, "stages": ["4"]},
    }
    
    # Analysis Settings
    TIPPING_POINT_STAGE: str = '4'
    DELTA_RANGE: List[float] = list(np.linspace(1.0, 4.0, 20))
    
    # Binary Classifier Parameters (for 88 classification)
    BINARY_CLASSIFIER_PARAMS: Dict[str, Any] = {
        'iterations': 1000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'task_type': 'GPU' if _GPU_INFO['available'] else 'CPU',
        'devices': '0' if _GPU_INFO['available'] else None,
        'random_seed': 42,
        'verbose': 100,
        'early_stopping_rounds': 50,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'thread_count': -1  # Use all CPU cores
    }
    
    # Update task type based on hardware
    if not _GPU_INFO['available']:
        BINARY_CLASSIFIER_PARAMS.pop('devices', None)  # Remove GPU-specific settings
    
    # Ordinal Classifier Parameters (for staging classification)
    ORDINAL_CLASSIFIER_PARAMS: Dict[str, Any] = {
        'iterations': 1000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'task_type': 'GPU' if _GPU_INFO['available'] else 'CPU',
        'devices': '0' if _GPU_INFO['available'] else None,
        'random_seed': 42,
        'verbose': 100,
        'early_stopping_rounds': 50,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'thread_count': -1  # Use all CPU cores
    }
    
    # Update task type based on hardware
    if not _GPU_INFO['available']:
        ORDINAL_CLASSIFIER_PARAMS.pop('devices', None)  # Remove GPU-specific settings

    @classmethod
    def get_output_prefix(cls) -> Path:
        """Generate output prefix with current timestamp."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return cls.OUTPUT_DIR / f"cancer_staging_imputation_{timestamp}"
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration settings."""
        if not 0 < cls.CONFIDENCE_LEVEL < 1:
            raise ValueError("CONFIDENCE_LEVEL must be between 0 and 1")
            
        if cls.N_IMPUTATIONS < 1:
            raise ValueError("N_IMPUTATIONS must be at least 1")
            
        # Validate CLASSES configuration
        if not hasattr(cls, 'CLASSES') or not cls.CLASSES:
            raise ValueError("CLASSES must be defined with at least one class")
            
        if not all(isinstance(class_name, str) for class_name in cls.CLASSES):
            raise ValueError("All elements in CLASSES must be strings")
            
        if not 0 < cls.SUBSAMPLE_RATIO <= 1:
            raise ValueError("SUBSAMPLE_RATIO must be between 0 and 1")
            
        if cls.K_PMM_NEIGHBORS < 1:
            raise ValueError("K_PMM_NEIGHBORS must be at least 1")
            
        if cls.RECIPIENT_QUERY_BATCH_SIZE < 1:
            raise ValueError("RECIPIENT_QUERY_BATCH_SIZE must be at least 1")
            
        # Validate hardware configuration
        if not isinstance(cls.USE_GPU, bool):
            raise ValueError("USE_GPU must be a boolean")
            
        if cls.DEVICE_TYPE not in ['cpu', 'gpu']:
            raise ValueError("DEVICE_TYPE must be either 'cpu' or 'gpu'")
            
        if cls.N_JOBS == 0 or cls.N_JOBS < -1:
            raise ValueError("N_JOBS must be -1 (all cores) or a positive integer")
            
        # Validate temperature scaling
        if not 0.1 <= cls.TEMPERATURE_SCALING <= 5.0:
            logging.warning(f"TEMPERATURE_SCALING ({cls.TEMPERATURE_SCALING}) is outside recommended range [0.1, 5.0]")
        
        # Validate classifier parameters
        for model_type, param_dict in [
            ('binary', cls.BINARY_CLASSIFIER_PARAMS),
            ('ordinal', cls.ORDINAL_CLASSIFIER_PARAMS)
        ]:
            if not isinstance(param_dict, dict):
                raise ValueError(f"{model_type.upper()}_CLASSIFIER_PARAMS must be a dictionary")
                
            if param_dict.get('iterations', 0) < 1:
                raise ValueError(f"{model_type} classifier: iterations must be at least 1")
                
            if not 0 < param_dict.get('learning_rate', 0.1) <= 1:
                raise ValueError(f"{model_type} classifier: learning_rate must be in (0, 1]")
                
            if param_dict.get('depth', 0) < 1:
                raise ValueError(f"{model_type} classifier: depth must be at least 1")
                
            # Check for GPU-specific settings
            if cls.USE_GPU:
                if param_dict.get('task_type') != 'GPU':
                    logging.warning(f"{model_type} classifier: task_type should be 'GPU' when USE_GPU is True")
            else:
                if param_dict.get('devices') is not None:
                    logging.warning(f"{model_type} classifier: devices should be None when not using GPU")
            
        # Validate paths
        # Check for required files
        if not cls.DATA_FILE.exists():
            logging.warning(f"Data file {cls.DATA_FILE} not found")
        if not cls.FEATURE_IMPORTANCE_FILE.exists():
            logging.warning(f"Feature importance file {cls.FEATURE_IMPORTANCE_FILE} not found")
        
        # Update subsample in classifier params to use the class attribute
        cls.BINARY_CLASSIFIER_PARAMS['subsample'] = cls.SUBSAMPLE_RATIO
        cls.ORDINAL_CLASSIFIER_PARAMS['subsample'] = cls.SUBSAMPLE_RATIO
        
        # Log successful validation
        logging.info("Configuration validation completed successfully")
    
    @classmethod
    def setup_logging(cls, output_prefix: Path = None) -> None:
        """Setup logging configuration."""
        # Use provided prefix or generate new one
        if output_prefix is None:
            output_prefix = cls.get_output_prefix()
        
        # Ensure the output directory exists
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Remove all handlers associated with the root logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        level = logging.DEBUG if cls.DEBUG else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{output_prefix}.log', mode='w'),
                logging.StreamHandler()
            ]
        )
        
        # Suppress matplotlib debug messages
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        
        return output_prefix

# Initialize and validate config
config = Config()
config.validate()
