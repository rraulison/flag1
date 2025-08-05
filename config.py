# this file handles the configuration of variables - config.py

"""
Enhanced configuration with better organization and validation.
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Union
import logging
from datetime import datetime

class Config:
    """Centralized configuration with validation."""
    
    # Determine the base directory of the project dynamically
    _BASE_DIR = Path(__file__).parent.parent.resolve()

    # Core Settings
    DEBUG: bool = True
    N_IMPUTATIONS: int = 5  # Reduced from 100 for testing
    CONFIDENCE_LEVEL: float = 0.95
    TARGET_VARIABLE: str = 'ESTADIAM'
    ORDINAL_TARGET_VARIABLE: str = 'estadiamento_ordinal'
    
    # Performance Settings
    RECIPIENT_QUERY_BATCH_SIZE: int = 10000  # Reduced batch size for better memory management
    SUBSAMPLE_RATIO: float = 0.8  # Slightly reduced for faster training
    EARLY_STOPPING_ROUNDS: int = 20  # Reduced for faster convergence
    K_PMM_NEIGHBORS: int = 5  # Reduced number of neighbors for faster PMM
    

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
    BINARY_CLASSIFIER_PARAMS = {
        "iterations": 2000,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_strength": 0.5,
        "min_data_in_leaf": 50,
        "border_count": 128,
        "grow_policy": "Depthwise",
        "subsample": 0.8,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "task_type": "GPU",
        "verbose": False,
        "allow_writing_files": False,
        "bootstrap_type": "Bernoulli",
        # Use all available CPU cores
        "thread_count": -1
    }
    
    # Ordinal Classifier Parameters (for staging classification)
    ORDINAL_CLASSIFIER_PARAMS = {
        "iterations": 2000,
        "depth": 8,
        "learning_rate": 0.075458,
        "l2_leaf_reg": 18.123750,
        "random_strength": 0.581402,
        "min_data_in_leaf": 97,
        "border_count": 84,
        "grow_policy": "Depthwise",
        "subsample": 0.654557,
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "task_type": "GPU",
        "verbose": False,
        "allow_writing_files": False,
        "bootstrap_type": "Bernoulli",
        # Use all available CPU cores
        "thread_count": -1
    }

    @classmethod
    def get_output_prefix(cls) -> Path:
        """Generate output prefix with current timestamp."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return cls.OUTPUT_DIR / f"cancer_staging_imputation_{timestamp}"
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration parameters and paths."""
        # Basic parameter validation
        assert 0 < cls.CONFIDENCE_LEVEL < 1, "Confidence level must be between 0 and 1"
        assert cls.N_IMPUTATIONS > 0, "Number of imputations must be positive"
        assert 0 < cls.SUBSAMPLE_RATIO <= 1, "Subsample ratio must be between 0 and 1"
        assert cls.K_PMM_NEIGHBORS > 0, "K neighbors must be positive"
        assert cls.RECIPIENT_QUERY_BATCH_SIZE > 0, "Batch size must be positive"
        
        # Validate classifier parameters
        for param_dict in [cls.BINARY_CLASSIFIER_PARAMS, cls.ORDINAL_CLASSIFIER_PARAMS]:
            assert param_dict.get('iterations', 0) > 0, "Iterations must be positive"
            assert 0 < param_dict.get('learning_rate', 0.1) <= 1, "Learning rate must be in (0, 1]"
            assert param_dict.get('depth', 0) > 0, "Depth must be positive"

        
        # Ensure required directories exist
        required_dirs = [cls.OUTPUT_DIR]
        for dir_path in required_dirs:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise RuntimeError(f"Failed to create directory {dir_path}: {e}")
        
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
