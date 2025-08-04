"""
This module defines the main pipeline for the imputation process with robust error handling.
"""
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd
from .config import Config
from .data_processing import DataProcessor
from .imputation import ImputationEngine
from .analysis import Analysis

logger = logging.getLogger(__name__)

class PipelineError(Exception):
    """Custom exception class for pipeline errors with context information."""
    def __init__(self, message: str, error_code: int = 1, context: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        context_str = ''
        if self.context:
            context_str = f'\nContext: ' + '; '.join(f'{k}={v}' for k, v in self.context.items())
        return f'{self.message}{context_str} (Error Code: {self.error_code})'

class Pipeline:
    """
    Orchestrates the entire imputation and analysis pipeline.
    """
    def __init__(self, config_obj: Config, output_prefix: Optional[Path] = None):
        """Initialize the pipeline with configuration and output settings.
        
        Args:
            config_obj: Configuration object with all necessary parameters
            output_prefix: Optional custom path prefix for output files
            
        Raises:
            PipelineError: If initialization fails due to invalid configuration or resources
        """
        try:
            self.config = config_obj
            self.output_prefix = output_prefix or config_obj.get_output_prefix()
            
            # Ensure output directory exists
            self.output_prefix.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize components with the output prefix
            logger.info("Initializing pipeline components...")
            self.data_processor = DataProcessor(self.config)
            self.imputation_engine = ImputationEngine(self.config)
            self.analysis = Analysis(self.config, self.output_prefix)
            
            logger.info(f"Pipeline initialized successfully. Output will be saved to: {self.output_prefix}")
            
        except Exception as e:
            error_context = {
                'output_prefix': str(output_prefix),
                'config_type': type(config_obj).__name__
            }
            raise PipelineError(
                f"Failed to initialize pipeline: {str(e)}",
                error_code=1001,
                context=error_context
            ) from e

    def _log_error_with_traceback(self, error: Exception, context: Optional[Dict] = None) -> None:
        """Log error with full traceback and context information.
        
        Args:
            error: The exception that was raised
            context: Additional context about where the error occurred
        """
        context = context or {}
        error_type = type(error).__name__
        error_message = str(error)
        
        # Log the full traceback to the error log
        error_trace = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        
        # Create a more readable error message
        context_str = '\n  '.join(f'{k}: {v}' for k, v in context.items())
        error_details = (
            f"\n{'='*80}\n"
            f"PIPELINE ERROR: {error_type}\n"
            f"Message: {error_message}\n"
            f"\nContext:\n  {context_str if context_str else 'None'}"
            f"\n\nFull traceback:\n{error_trace}"
            f"{'='*80}"
        )
        
        logger.error(error_details)
        
        # Also write to a dedicated error log file
        error_log_path = self.output_prefix.parent / f"{self.output_prefix.name}_error.log"
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*50} {pd.Timestamp.now()} {'='*50}\n")
            f.write(error_details)
        
        logger.info(f"Full error details written to: {error_log_path}")

    def run(self) -> Dict[str, Any]:
        """
        Executes the full pipeline including data loading, imputation, analysis, and reporting.
        
        Returns:
            Dictionary containing:
            - status: 'success' or 'error'
            - imputed_data: The final imputed dataset (if successful)
            - results_summary: Summary of imputation results
            - final_results: Formatted results table
            - model_performance: Performance metrics of the models
            - error: Error message (if status is 'error')
            - traceback: Full traceback (if status is 'error')
            
        Raises:
            PipelineError: If a critical error occurs during pipeline execution
        """
        logger.info("=" * 80)
        logger.info(f"STARTING PIPELINE EXECUTION - {pd.Timestamp.now()}")
        logger.info("=" * 80)
        
        # Track the current step for better error reporting
        current_step = "initialization"
        
        try:
            # 1. Load and preprocess data
            current_step = "data_loading"
            logger.info("\n" + "="*50)
            logger.info("STEP 1/4: LOADING AND PREPROCESSING DATA")
            logger.info("="*50)
            
            df_filt, feature_cols, cat_idx = self.data_processor.load_and_preprocess_data()
            logger.info(f"✓ Data loaded and preprocessed successfully. Shape: {df_filt.shape}")
            logger.info(f"  Features: {len(feature_cols)} total, {len(cat_idx)} categorical")
            
            # 2. Run imputation
            current_step = "imputation"
            logger.info("\n" + "="*50)
            logger.info("STEP 2/4: RUNNING IMPUTATION")
            logger.info("="*50)
            
            scenario_proportions, last_imputed_dataset, model_performance, last_model_artifacts = \
                self.imputation_engine.run_imputation_loop(df_filt, feature_cols, cat_idx)
            
            logger.info("✓ Imputation process completed successfully.")
            logger.info(f"  Final imputed dataset shape: {last_imputed_dataset.shape}")
            
            # 3. Analyze results
            current_step = "analysis"
            logger.info("\n" + "="*50)
            logger.info("STEP 3/4: ANALYZING RESULTS")
            logger.info("="*50)
            
            results_summary, pooled_metrics, scenario_names, original_props = self.analysis.analyze_results(
                scenario_proportions=scenario_proportions,
                imputed_dataset=last_imputed_dataset,
                original_dataset=df_filt.copy(),  # Pass original df for comparison
                model_performance=model_performance,
                model_artifacts=last_model_artifacts
            )
            
            # Store pooled_metrics as an instance variable for diagnostics
            self.analysis.pooled_metrics = pooled_metrics
            logger.info("✓ Results analysis completed successfully.")

            # 4. Generate and print summary table
            current_step = "summary_generation"
            logger.info("\n" + "="*50)
            logger.info("STEP 4/4: GENERATING SUMMARY AND REPORTS")
            logger.info("="*50)
            
            final_results = self.analysis.generate_summary_table(
                pooled_metrics=pooled_metrics,
                original_props=original_props,
                classes_=list(original_props.index)
            )
            logger.info("\n" + str(final_results))

            # 5. Run diagnostics
            current_step = "diagnostics"
            logger.info("\nRunning diagnostics...")
            self.analysis.run_diagnostics(
                model_performance=model_performance,
                model_artifacts=last_model_artifacts,
                imputed_dataset=last_imputed_dataset,
                original_dataset=df_filt.copy()  # Pass original df for comparison
            )

            # 6. Create visualizations
            current_step = "visualization"
            logger.info("\nGenerating visualizations...")
            if hasattr(self.analysis, 'create_visualizations'):
                self.analysis.create_visualizations(
                    results_summary=results_summary,
                    scenario_proportions=scenario_proportions,
                    scenario_names=scenario_names,
                    classes_=list(original_props.index)
                )

            # 7. Generate and export report if the method exists
            current_step = "report_generation"
            if hasattr(self.analysis, 'generate_and_export_report'):
                logger.info("\nGenerating and exporting report...")
                try:
                    # Get known data (non-imputed) for reporting
                    target_var = self.config.TARGET_VARIABLE
                    df_known = df_filt[df_filt[target_var] != '99'].copy()
                    
                    self.analysis.generate_and_export_report(
                        df_original=df_filt,
                        df_known=df_known,
                        df_imputed=last_imputed_dataset,
                        results_summary=results_summary,
                        model_performance=model_performance,
                        final_results=final_results,
                        classes_=list(original_props.index)
                    )
                except Exception as e:
                    self._log_error_with_traceback(e, {
                        'step': 'report_generation',
                        'error_type': type(e).__name__
                    })
                    # Continue execution even if report generation fails

            logger.info("\n" + "="*80)
            logger.info(f"PIPELINE COMPLETED SUCCESSFULLY - {pd.Timestamp.now()}")
            logger.info("="*80)
            
            return {
                'status': 'success',
                'imputed_data': last_imputed_dataset,
                'results_summary': results_summary,
                'final_results': final_results,
                'model_performance': model_performance
            }

        except Exception as e:
            error_context = {
                'step': current_step,
                'error_type': type(e).__name__,
                'output_prefix': str(self.output_prefix)
            }
            
            # Log the error with full traceback
            self._log_error_with_traceback(e, error_context)
            
            # Re-raise as a PipelineError with context
            raise PipelineError(
                f"Pipeline failed during {current_step.replace('_', ' ')}: {str(e)}",
                error_code=2000 + len(current_step),  # Generate unique error codes
                context=error_context
            ) from e
