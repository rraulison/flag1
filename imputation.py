"""
Enhanced imputation logic with checkpointing, better error handling, and monitoring.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier
from pathlib import Path
from typing import Any, Dict, List, Tuple
import gc
import json
import pickle
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import os
from contextlib import contextmanager
import time

# Configure logger
logger = logging.getLogger(__name__)

# Import functions from utils.py
from .utils import safe_pmm_imputation, apply_delta_adjustment, predict_proba_in_batches
from .config import config

@contextmanager
def imputation_timer(iteration: int):
    """Context manager for timing imputation iterations."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logging.debug(f"Imputation {iteration} completed in {elapsed:.2f}s")

class ImputationEngine:
    """Main imputation engine for two-step imputation."""

    def __init__(self, config):
        self.config = config
        self.model_performance = {'binary': [], 'ordinal': []}
        self.last_model_artifacts = {}
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir = Path(self.config.CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, iteration: int, imputation_results: List[pd.DataFrame]) -> None:
        """
        Save checkpoint with atomic write to prevent corruption.
        
        Args:
            iteration: Current iteration number
            imputation_results: List of DataFrames with imputation results
        """
        checkpoint_path = self.checkpoint_dir / f'imputation_checkpoint_{iteration}.pkl'
        temp_path = checkpoint_path.with_suffix('.tmp')
        
        state = {
            'iteration': iteration,
            'model_performance': self.model_performance,
            'last_model_artifacts': self.last_model_artifacts,
            'imputation_results': imputation_results
        }
        
        try:
            # Write to temporary file first
            with open(temp_path, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename (POSIX operation)
            temp_path.replace(checkpoint_path)
            logger.info(f"Successfully saved checkpoint for iteration {iteration}")
            
        except Exception as e:
            # Clean up temporary file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up temporary checkpoint file: {cleanup_error}")
            
            logger.error(f"Failed to save checkpoint for iteration {iteration}: {e}")
            raise
    
    def _load_checkpoint(self, iteration: int) -> Optional[dict]:
        """Load a previously saved checkpoint."""
        checkpoint_path = self.checkpoint_dir / f'imputation_checkpoint_{iteration}.pkl'
        if checkpoint_path.exists():
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _cleanup_checkpoints(self):
        """Remove checkpoint files after successful completion."""
        for file in self.checkpoint_dir.glob('imputation_checkpoint_*.pkl'):
            try:
                file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {file}: {e}")
    
    def run_imputation_loop(
        self, df: pd.DataFrame, feature_cols: List[str], cat_idx: List[int]
    ) -> Tuple[Dict, pd.DataFrame, Dict, Dict]:
        """
        Orchestrates the two-step imputation process over N iterations with checkpointing.
        
        Args:
            df: Input DataFrame with the data to be imputed
            feature_cols: List of feature column names
            cat_idx: List of indices for categorical features
            
        Returns:
            Tuple containing:
                - scenario_proportions: Dictionary of proportions for each scenario
                - final_imputed_dataset: The final imputed dataset
                - model_performance: Dictionary of model performance metrics
                - last_model_artifacts: Artifacts from the last model
        """
        logger.info(f"Starting imputation loop with {self.config.N_IMPUTATIONS} iterations.")
        
        # Check for existing checkpoints
        last_checkpoint = None
        for i in range(self.config.N_IMPUTATIONS - 1, -1, -1):
            checkpoint = self._load_checkpoint(i)
            if checkpoint:
                last_checkpoint = checkpoint
                logger.info(f"Resuming from checkpoint at iteration {i+1}")
                break
        
        if last_checkpoint:
            # Restore state from checkpoint
            start_iter = last_checkpoint['iteration'] + 1
            imputation_results = last_checkpoint['imputation_results']
            self.model_performance = last_checkpoint['model_performance']
            self.last_model_artifacts = last_checkpoint['last_model_artifacts']
        else:
            start_iter = 0
            imputation_results = []
        
        try:
            for i in range(start_iter, self.config.N_IMPUTATIONS):
                start_time = time.time()
                logger.info(f"Starting imputation {i+1}/{self.config.N_IMPUTATIONS}")
                
                try:
                    with imputation_timer(i + 1):
                        imputed_df_single_run = self._run_single_imputation(
                            df, feature_cols, cat_idx, i
                        )
                        imputation_results.append(imputed_df_single_run)
                        
                        # Save checkpoint if enabled
                        if self.config.CHECKPOINT_INTERVAL > 0 and \
                           (i + 1) % self.config.CHECKPOINT_INTERVAL == 0:
                            self._save_checkpoint(i, imputation_results)
                        
                        iteration_time = time.time() - start_time
                        logger.info(f"Completed imputation {i+1}/{self.config.N_IMPUTATIONS} "
                                  f"in {iteration_time:.2f} seconds")
                        
                        # Estimate time remaining
                        remaining_iterations = self.config.N_IMPUTATIONS - (i + 1)
                        if remaining_iterations > 0:
                            avg_time = (time.time() - start_time) / (i + 1 - start_iter)
                            remaining_time = avg_time * remaining_iterations
                            logger.info(f"Estimated time remaining: {remaining_time/60:.1f} minutes")
                
                except Exception as e:
                    logger.error(f"Imputation {i + 1} failed: {e}", exc_info=True)
                    # Save failed state for debugging
                    self._save_checkpoint(f"failed_{i}", imputation_results)
                    raise
            
            logger.info("Finished all imputation iterations.")
            
            # Clean up checkpoints on successful completion
            if not self.config.DEBUG:
                self._cleanup_checkpoints()
            
            # Return the final results
            final_imputed_dataset = imputation_results[-1]
            scenario_proportions = self._aggregate_proportions(imputation_results)
            
            return scenario_proportions, final_imputed_dataset, self.model_performance, self.last_model_artifacts
            
        except Exception as e:
            logger.error(f"Imputation loop failed: {e}", exc_info=True)
            raise

    def _run_single_imputation(
        self, df: pd.DataFrame, feature_cols: List[str], cat_idx: List[int], iteration: int
    ) -> pd.DataFrame:
        """Executes a single two-step imputation: flag imputation then ordinal imputation."""
        
        df_known = df[df[self.config.TARGET_VARIABLE] != '99'].copy()
        df_boot = df_known.sample(n=len(df_known), replace=True, random_state=iteration)

        df_imputed_flag = self._impute_flag(df.copy(), df_boot, feature_cols, cat_idx, iteration)

        df_imputed_ordinal = self._impute_ordinal(df_imputed_flag, df_boot, feature_cols, cat_idx, iteration)
        
        df_reconstructed = self._reconstruct_staging(df_imputed_ordinal)
        
        return df_reconstructed

    def _reconstruct_staging(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reconstructs the final 'ESTADIAM' column after imputation."""
        df_final = df.copy()
        
        # Update ESTADIAM based on the imputed ordinal and flag variables
        # Condition 1: If flag is 1, ESTADIAM is '88'
        df_final.loc[df_final['flag_nao_estadiavel'] == 1, self.config.TARGET_VARIABLE] = '88'
        
        # Condition 2: If flag is 0 and ordinal is not null, map ordinal back to ESTADIAM
        ordinal_not_null = df_final[self.config.ORDINAL_TARGET_VARIABLE].notna()
        flag_is_zero = df_final['flag_nao_estadiavel'] == 0
        
        df_final.loc[flag_is_zero & ordinal_not_null, self.config.TARGET_VARIABLE] = \
            df_final.loc[flag_is_zero & ordinal_not_null, self.config.ORDINAL_TARGET_VARIABLE].astype(int).astype(str)
            
        logger.info("Reconstructed 'ESTADIAM' column from imputed values.")
        logger.debug(f"Final ESTADIAM distribution:\n{df_final[self.config.TARGET_VARIABLE].value_counts(dropna=False)}")
        
        return df_final

    def _impute_flag(self, df_full: pd.DataFrame, df_boot: pd.DataFrame, feature_cols: List[str], cat_idx: List[int], iteration: int) -> pd.DataFrame:
        """
        Imputes the 'flag_nao_estadiavel' for unknown cases ('99').
        
        This is the first step of the two-step imputation process:
        1. A binary classification model is trained on a bootstrap sample of known data
           to distinguish between 'estadiável' (0) and 'não estadiável' (1).
        2. The trained model predicts the probability of being 'não estadiável' for all
           unknown cases (where ESTADIAM was '99').
        3. A Bernoulli draw is performed for each unknown case based on its predicted
           probability to assign it as either 0 or 1, capturing the uncertainty.
        
        Args:
            df_full: Full DataFrame containing all data.
            df_boot: Bootstrap sample of the data for model training.
            feature_cols: List of feature column names.
            cat_idx: List of indices for categorical features.
            iteration: Current imputation iteration number.
            
        Returns:
            DataFrame with imputed values for flag_nao_estadiavel.
        """
        target_flag = 'flag_nao_estadiavel'
        
        # Define staging variables that should not be used as features
        staging_vars = [
            'TNM', 'OUTROESTA', 'T', 'N', 'M', 'pT', 'pN', 'pM', 'PTNM', 'ESTDFIMT',
            self.config.TARGET_VARIABLE, self.config.ORDINAL_TARGET_VARIABLE,
            target_flag, 'flag_nao_estadiavel'
        ]
        
        # Exclude target and staging variables from features
        features_to_use = [f for f in feature_cols if f not in staging_vars]
        
        logger.info(f"[Binary Model] Training with {len(features_to_use)} features: {features_to_use}")
        logger.debug(f"Excluded staging variables: {[f for f in staging_vars if f in feature_cols]}")
        
        # Recalculate categorical indices for the filtered features
        # Map original feature indices to their positions in the filtered list
        feature_to_new_idx = {f: i for i, f in enumerate(features_to_use)}
        
        # Filter and remap categorical indices
        filtered_cat_idx = [
            feature_to_new_idx[feature_cols[idx]] 
            for idx in cat_idx 
            if feature_cols[idx] in feature_to_new_idx
        ]
        
        logger.debug(f"Original cat_idx: {cat_idx}")
        logger.debug(f"Filtered cat_idx: {filtered_cat_idx}")
        
        # Prepare training data - only use rows with non-missing flag values
        train_mask = df_boot['flag_nao_estadiavel'].notna()
        X_train = df_boot.loc[train_mask, features_to_use]
        y_train = df_boot.loc[train_mask, target_flag].astype(int)
        
        # Train the binary classification model with filtered categorical indices
        model, val_logloss = self._train_model(X_train, y_train, filtered_cat_idx, iteration, 'binary')
        self.model_performance['binary'].append(val_logloss)
        
        # Identify rows that need imputation
        to_impute_mask = df_full[target_flag].isna()
        X_to_impute = df_full.loc[to_impute_mask, feature_cols]
        
        if not X_to_impute.empty:
            # Get predicted probabilities for missing values
            pred_probs = predict_proba_in_batches(
                model, 
                X_to_impute[features_to_use], 
                self.config.RECIPIENT_QUERY_BATCH_SIZE
            )
            
            # Draw from Bernoulli distribution using predicted probabilities
            # This better captures uncertainty than thresholding at 0.5
            rng = np.random.RandomState(seed=42 + iteration)  # Ensure reproducibility
            imputed_flags = rng.binomial(1, pred_probs[:, 1]).astype(int)
            
            # Update the original DataFrame with imputed values
            df_full.loc[to_impute_mask, target_flag] = imputed_flags
            
            # Log some statistics about the imputation
            n_imputed = to_impute_mask.sum()
            pct_positive = imputed_flags.mean() * 100
            logger.info(
                f"Imputed {n_imputed} missing flags: "
                f"{pct_positive:.1f}% as 'não estadiável' (1), "
                f"{100 - pct_positive:.1f}% as 'estadiável' (0)"
            )

        # Save the model from the last iteration for potential later use
        if iteration == self.config.N_IMPUTATIONS - 1:
            self.last_model_artifacts['binary_model'] = model
            
            # Log feature importance for the final model
            if hasattr(model, 'feature_importances_'):
                try:
                    # Get the actual feature names used by the model
                    if hasattr(model, 'feature_names_'):
                        # If model has feature_names_ attribute, use it to ensure correct alignment
                        model_feature_names = model.feature_names_
                        feature_importance = pd.Series(
                            model.feature_importances_,
                            index=model_feature_names
                        ).sort_values(ascending=False)
                    else:
                        # Fallback to provided feature_cols, but ensure lengths match
                        if len(model.feature_importances_) == len(feature_cols):
                            feature_importance = pd.Series(
                                model.feature_importances_,
                                index=feature_cols
                            ).sort_values(ascending=False)
                        else:
                            # If lengths still don't match, use numeric indices
                            logger.warning(
                                f"Feature importance length ({len(model.feature_importances_)}) "
                                f"does not match feature columns length ({len(feature_cols)}). "
                                "Using numeric indices."
                            )
                            feature_importance = pd.Series(
                                model.feature_importances_
                            ).sort_values(ascending=False)
                    
                    logger.info("Top 10 most important features for flag imputation:")
                    logger.info(feature_importance.head(10).to_string())
                    
                except Exception as e:
                    logger.error(f"Error logging feature importance: {e}", exc_info=True)

        return df_full

    def _impute_ordinal(self, df_full: pd.DataFrame, df_boot: pd.DataFrame, feature_cols: List[str], cat_idx: List[int], iteration: int) -> pd.DataFrame:
        """
        Imputes the ordinal staging value for cases determined to be 'estadiável'.

        This is the second step of the two-step imputation process:
        1. An ordinal classification model is trained on a bootstrap sample of known,
           'estadiável' data.
        2. The model predicts staging probabilities for cases that were originally unknown
           but were imputed as 'estadiável' (flag_nao_estadiavel = 0) in the first step.
        3. Predictive Mean Matching (PMM) is used to impute the final staging value.
           This involves finding observed cases (donors) with similar predicted probabilities
           to the case being imputed (recipient) and randomly selecting one of their
           actual staging values. This preserves the original data distribution.

        Args:
            df_full: Full DataFrame with imputed flags.
            df_boot: Bootstrap sample for model training.
            feature_cols: List of feature column names.
            cat_idx: List of indices for categorical features.
            iteration: Current imputation iteration number.

        Returns:
            DataFrame with imputed ordinal staging values.
        """
        target_ordinal = self.config.ORDINAL_TARGET_VARIABLE
        
        # Prepare data for ordinal imputation (only rows where flag_nao_estadiavel == 0)
        df_estadiavel = df_full[df_full['flag_nao_estadiavel'] == 0].copy()
        df_boot_estadiavel = df_boot[df_boot['flag_nao_estadiavel'] == 0].copy()
        
        # Define target variable and ensure we exclude it from features
        target_ordinal = self.config.ORDINAL_TARGET_VARIABLE
        
        # Define staging variables that should not be used as features
        staging_vars = [
            'TNM', 'OUTROESTA', 'T', 'N', 'M', 'pT', 'pN', 'pM', 'PTNM', 'ESTDFIMT',
            self.config.TARGET_VARIABLE, target_ordinal,
            'flag_nao_estadiavel'
        ]
        
        # Exclude target and staging variables from features
        features_to_use = [f for f in feature_cols if f not in staging_vars]
        
        logger.info(f"[Ordinal Model] Training with {len(features_to_use)} features: {features_to_use}")
        logger.debug(f"Excluded staging variables: {[f for f in staging_vars if f in feature_cols]}")
        
        # Recalculate categorical indices for the filtered features
        # Map original feature indices to their positions in the filtered list
        feature_to_new_idx = {f: i for i, f in enumerate(features_to_use)}
        
        # Filter and remap categorical indices
        filtered_cat_idx = [
            feature_to_new_idx[feature_cols[idx]] 
            for idx in cat_idx 
            if feature_cols[idx] in feature_to_new_idx
        ]
        
        logger.debug(f"[Ordinal Model] Original cat_idx: {cat_idx}")
        logger.debug(f"[Ordinal Model] Filtered cat_idx: {filtered_cat_idx}")
        
        # Get indices where we have both features and target values
        valid_indices = df_boot_estadiavel[target_ordinal].dropna().index.intersection(
            df_boot_estadiavel[features_to_use].dropna(how='any').index
        )
        
        if len(valid_indices) == 0:
            logger.warning("No valid training samples with both features and target values")
            return df_full
            
        # Split into features and target using aligned indices
        X_train = df_boot_estadiavel.loc[valid_indices, features_to_use]
        y_train = df_boot_estadiavel.loc[valid_indices, target_ordinal].astype(int)
        
        # Log class distribution for debugging
        class_dist = y_train.value_counts().sort_index()
        logger.debug(f"Training ordinal model with {len(X_train)} samples. Class distribution:\n{class_dist}")
        
        # Train the ordinal classification model with filtered categorical indices
        model, val_logloss = self._train_model(X_train, y_train, filtered_cat_idx, iteration, 'ordinal')
        self.model_performance['ordinal'].append(val_logloss)
        
        # Identify rows that need imputation (flag=0 and ordinal is missing)
        to_impute_mask = (df_full['flag_nao_estadiavel'] == 0) & (df_full[target_ordinal].isna())
        X_to_impute = df_full.loc[to_impute_mask, feature_cols]
        
        if not X_to_impute.empty:
            # Get predicted probabilities for recipients (missing values)
            pred_probs_recipients = predict_proba_in_batches(
                model, 
                X_to_impute[features_to_use], 
                self.config.RECIPIENT_QUERY_BATCH_SIZE
            )
            
            # Get predicted probabilities for donors (known values in bootstrap sample)
            pred_probs_donors = model.predict_proba(X_train)
            known_donors_labels = y_train.astype(int) 
            
            # Ensure we have valid donors
            if len(known_donors_labels) == 0:
                logger.warning("No valid donors available for PMM. Using mode imputation.")
                mode_value = y_train.mode().iloc[0] if not y_train.empty else 0
                imputed_values = np.full(len(X_to_impute), mode_value)
            else:
                # Use PMM to find the best matching donors
                imputed_values = safe_pmm_imputation(
                    pred_probs_recipients,  # Shape: (n_recipients, n_classes)
                    pred_probs_donors,      # Shape: (n_donors, n_classes)
                    known_donors_labels,    # Shape: (n_donors,)
                    k_neighbors=min(self.config.K_PMM_NEIGHBORS, len(known_donors_labels)),
                    random_state=42 + iteration  # For reproducibility
                )
            
            if imputed_values is not None:
                # Apply the imputed values
                df_full.loc[to_impute_mask, target_ordinal] = imputed_values
                
                # Log imputation statistics
                n_imputed = to_impute_mask.sum()
                imputed_dist = pd.Series(imputed_values).value_counts().sort_index()
                logger.info(
                    f"Imputed {n_imputed} missing ordinal values. "
                    f"Distribution: {imputed_dist.to_dict()}"
                )
                
                # For the first iteration, log more detailed information
                if iteration == 0:
                    logger.debug(f"Example of imputed values: {imputed_values[:10]}...")
                    logger.debug(f"Donor class distribution: {pd.Series(known_donors_labels).value_counts().sort_index().to_dict()}")
        else:
            logger.info("No ordinal values to impute in this iteration.")
        
        # Save the model from the last iteration for potential later use
        if iteration == self.config.N_IMPUTATIONS - 1:
            self.last_model_artifacts['ordinal_model'] = model
            # CRITICAL FIX: Save required artifacts for PMM and MNAR analysis
            try:
                self.last_model_artifacts['p_recipients'] = pred_probs_recipients
                self.last_model_artifacts['p_donors'] = pred_probs_donors  
                self.last_model_artifacts['y_donors_iter'] = known_donors_labels
            except NameError:
                logger.warning("Could not save PMM artifacts - variables not in scope")
            
            # Log feature importance for the final model
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.Series(
                    model.feature_importances_, 
                    index=features_to_use  # FIXED: Use features actually used in training
                ).sort_values(ascending=False)
                logger.debug("Top 10 important features for ordinal imputation:")
                logger.debug(feature_importance.head(10).to_string())
        
        return df_full

    def _train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, 
        cat_idx: List[int], iteration: int, model_type: str
    ) -> Tuple[Any, float]:
        """Trains a CatBoost model (binary or ordinal) with improved validation."""
        # Get base parameters
        params = (
            self.config.BINARY_CLASSIFIER_PARAMS if model_type == 'binary' 
            else self.config.ORDINAL_CLASSIFIER_PARAMS
        ).copy()
        
        if y_train.nunique() < 2:
            raise ValueError(f"Training data for {model_type} model has only one class.")

        # FIXED: Proper class weight handling for both model types
        if model_type == 'binary':
            # Use CatBoost's auto_class_weights for better balance
            params['auto_class_weights'] = 'Balanced'
        else:
            # For ordinal, use scale_pos_weight or manual class weights
            class_counts = y_train.value_counts().sort_index()
            total_samples = len(y_train)
            # Calculate inverse frequency weights
            class_weights = {}
            for cls in class_counts.index:
                class_weights[cls] = total_samples / (len(class_counts) * class_counts[cls])
            params['class_weights'] = class_weights
        
        # Split data with stratification to maintain class distribution
        X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(
            X_train, y_train, 
            test_size=0.2, 
            random_state=42 + iteration,  # More stable random state
            stratify=y_train
        )

        # Add early stopping and reduce overfitting
        if 'early_stopping_rounds' not in params:
            params['early_stopping_rounds'] = 50
        if 'learning_rate' not in params:
            params['learning_rate'] = 0.05
        if 'depth' not in params:
            params['depth'] = 6  # Shallower trees to prevent overfitting
        if 'l2_leaf_reg' not in params:
            params['l2_leaf_reg'] = 3.0  # Add L2 regularization
        
        model = CatBoostClassifier(**params)
        
        # Fit with more detailed logging
        model.fit(
            X_train_part, y_train_part,
            eval_set=(X_val_part, y_val_part),
            cat_features=cat_idx,
            verbose=100,  # Show progress every 100 iterations
            plot=False
        )

        # Get predicted probabilities with temperature scaling to avoid overconfidence
        val_preds = model.predict_proba(X_val_part)
        
        # Apply temperature scaling to soften probabilities
        temperature = 0.5  # Adjust this value as needed
        val_preds = np.power(val_preds, 1.0/temperature)
        val_preds = val_preds / val_preds.sum(axis=1, keepdims=1)
        
        # Log model and data information
        logger.debug(f"Model type: {model_type}")
        logger.debug(f"Model classes: {model.classes_}")
        logger.debug(f"Unique y_train values: {np.unique(y_train_part)}")
        logger.debug(f"Unique y_val values: {np.unique(y_val_part)}")
        logger.debug(f"Class distribution - Train: {dict(y_train_part.value_counts())}")
        logger.debug(f"Class distribution - Val: {dict(y_val_part.value_counts())}")
        
        # Log feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,  # Use actual training features
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            logger.debug(f"Top 10 important features for {model_type} model:")
            logger.debug(feature_importance.head(10).to_string())
        
        # Ensure y_val_part is numeric for log_loss calculation
        y_val_numeric = pd.Categorical(y_val_part, categories=model.classes_).codes
        
        # Calculate metrics
        try:
            val_logloss = log_loss(
                y_val_numeric, 
                val_preds,
                labels=np.arange(len(model.classes_)),
                sample_weight=None
            )
            
            # Additional metrics
            accuracy = accuracy_score(y_val_part, model.predict(X_val_part))
            logger.info(
                f"{model_type.capitalize()} model - Iter {iteration + 1} - "
                f"Val LogLoss: {val_logloss:.4f} - "
                f"Accuracy: {accuracy:.4f} - "
                f"Classes: {model.classes_}"
            )
            
            # Log feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                logger.debug(f"Top 10 important features:\n{feature_importance.head(10)}")
                
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            logger.error(f"y_val_numeric shape: {y_val_numeric.shape}, unique: {np.unique(y_val_numeric)}")
            logger.error(f"val_preds shape: {val_preds.shape}, sum: {np.sum(val_preds, axis=1)[:10]}")
            val_logloss = 1.0  # Penalize failed evaluations

        return model, val_logloss

    def _aggregate_proportions(self, imputation_results: List[pd.DataFrame]) -> Dict:
        """Aggregates proportions from multiple imputation runs."""
        all_proportions = []
        for run_df in imputation_results:
            # Calculate proportions from the final, reconstructed ESTADIAM column
            proportions = run_df[self.config.TARGET_VARIABLE].value_counts(normalize=True)
            all_proportions.append(proportions)
        
        # The structure is kept to be compatible with the analysis module
        return {'mar': all_proportions} 

