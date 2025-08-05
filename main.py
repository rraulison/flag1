#this file handles the main pipeline - main.py
"""
Main script to run the revised imputation pipeline (v6).
"""
import logging
from flag1.config import config
from flag1.pipeline import Pipeline

logger = logging.getLogger(__name__)

def main():
    """Main function to run the pipeline."""
    # Setup logging with dynamic output prefix
    output_prefix = config.setup_logging()
    
    # Create pipeline with the output prefix
    pipeline = Pipeline(config, output_prefix)
    pipeline.run()

if __name__ == '__main__':
    main()