# this file handles the imputation of missing values - imputation.py
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
        
        # Initialize model performance tracking
        self.model_performance = {'binary': [], 'ordinal': []}
        self.last_model_artifacts = {}
    
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
        
        # Initialize results storage
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
                    raise
            
            logger.info("Finished all imputation iterations.")
            
            # Return the final results
            if not imputation_results:
                raise ValueError("No imputation results were generated. Check for errors in the imputation process.")
                
            final_imputed_dataset = imputation_results[-1]
            if not isinstance(final_imputed_dataset, dict):
                raise TypeError(f"Expected final_imputed_dataset to be a dict, got {type(final_imputed_dataset).__name__}")
                
            scenario_proportions = self._aggregate_proportions(imputation_results)
            
            # Ensure MAR results exist
            if 'mar' not in final_imputed_dataset:
                raise KeyError("MAR scenario not found in imputation results")
                
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

        df_imputed_ordinal_mar, all_imputed_values = self._impute_ordinal(
            df_imputed_flag, df_boot, feature_cols, cat_idx, iteration
        )
        
        all_reconstructed_dfs = {}

        # Reconstruct MAR scenario from the main dataframe returned by _impute_ordinal
        df_reconstructed_mar = self._reconstruct_staging(df_imputed_ordinal_mar.copy())
        all_reconstructed_dfs['mar'] = df_reconstructed_mar

        # Reconstruct MNAR scenarios using the dictionary of imputed values
        # The flag-imputed dataframe is the base for all scenarios
        flag_zero = df_imputed_flag['flag_nao_estadiavel'] == 0
        ordinal_na = df_imputed_flag[self.config.ORDINAL_TARGET_VARIABLE].isna()
        to_impute_mask = flag_zero & ordinal_na

        for scenario_name, imputed_values in all_imputed_values.items():
            if scenario_name == 'mar':
                continue  # Already handled

            df_scenario = df_imputed_flag.copy()
            df_scenario.loc[to_impute_mask, self.config.ORDINAL_TARGET_VARIABLE] = imputed_values
            
            df_reconstructed_scenario = self._reconstruct_staging(df_scenario)
            all_reconstructed_dfs[scenario_name] = df_reconstructed_scenario
            
        return all_reconstructed_dfs

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
                        # Fallback to features_to_use, ensuring lengths match
                        if len(model.feature_importances_) == len(features_to_use):
                            feature_importance = pd.Series(
                                model.feature_importances_,
                                index=features_to_use
                            ).sort_values(ascending=False)
                        else:
                            # If lengths still don't match, use numeric indices as a last resort
                            logger.warning(
                                f"Feature importance length ({len(model.feature_importances_)}) "
                                f"does not match feature columns length ({len(features_to_use)}). "
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

    def _impute_ordinal(self, df_full: pd.DataFrame, df_boot: pd.DataFrame, feature_cols: List[str], cat_idx: List[int], iteration: int) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
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
        4. For MNAR scenarios, it applies a delta adjustment before PMM.

        Args:
            df_full: Full DataFrame with imputed flags.
            df_boot: Bootstrap sample for model training.
            feature_cols: List of feature column names.
            cat_idx: List of indices for categorical features.
            iteration: Current imputation iteration number.

        Returns:
            A tuple containing:
            - DataFrame with imputed ordinal staging values for the MAR scenario.
            - Dictionary with imputed values for each MNAR scenario.
        """
        target_ordinal = self.config.ORDINAL_TARGET_VARIABLE
        
        # Prepare data for ordinal imputation (only rows where flag_nao_estadiavel == 0)
        df_estadiavel = df_full[df_full['flag_nao_estadiavel'] == 0].copy()
        df_boot_estadiavel = df_boot[df_boot['flag_nao_estadiavel'] == 0].copy()
        
        # Define target variable and ensure we exclude it from features
        target_ordinal = self.config.ORDINAL_TARGET_VARIABLE
        
        # Define staging variables that should NOT be used as features
        # Keep the user-requested staging variables as features
        staging_vars_to_exclude = [
            self.config.TARGET_VARIABLE, 
            target_ordinal,
            'flag_nao_estadiavel'
        ]
        
        # Exclude only the specified target and flag variables from features
        features_to_use = [f for f in feature_cols if f not in staging_vars_to_exclude]
        
        logger.info(f"[Ordinal Model] Training with {len(features_to_use)} features.")
        logger.debug(f"Features used: {features_to_use}")
        logger.debug(f"Excluded variables: {staging_vars_to_exclude}")
        
        # Recalculate categorical indices for the filtered features
        feature_to_new_idx = {f: i for i, f in enumerate(features_to_use)}
        filtered_cat_idx = [
            feature_to_new_idx[feature_cols[idx]] 
            for idx in cat_idx 
            if feature_cols[idx] in feature_to_new_idx
        ]
        
        # Get indices where we have both features and target values
        valid_indices = df_boot_estadiavel[target_ordinal].dropna().index.intersection(
            df_boot_estadiavel[features_to_use].dropna(how='any').index
        )
        
        if len(valid_indices) == 0:
            logger.warning("No valid training samples for ordinal model; returning original data.")
            return df_full, {}
            
        X_train = df_boot_estadiavel.loc[valid_indices, features_to_use]
        y_train = df_boot_estadiavel.loc[valid_indices, target_ordinal].astype(int)
        
        class_dist = y_train.value_counts().sort_index()
        logger.debug(f"Training ordinal model with {len(X_train)} samples. Class distribution:\n{class_dist}")
        
        model, val_logloss = self._train_model(X_train, y_train, filtered_cat_idx, iteration, 'ordinal')
        self.model_performance['ordinal'].append(val_logloss)
        
        to_impute_mask = (df_full['flag_nao_estadiavel'] == 0) & (df_full[target_ordinal].isna())
        X_to_impute = df_full.loc[to_impute_mask, features_to_use]

        all_imputed_values = {}
        df_full_mar = df_full.copy() # Create a copy for the main MAR imputation

        if not X_to_impute.empty:
            pred_probs_recipients = predict_proba_in_batches(
                model, X_to_impute, self.config.RECIPIENT_QUERY_BATCH_SIZE
            )
            pred_probs_donors = model.predict_proba(X_train)
            known_donors_labels = y_train.astype(int)

            if len(known_donors_labels) == 0:
                logger.warning("No valid donors for PMM. Using mode imputation for all scenarios.")
                mode_value = y_train.mode().iloc[0] if not y_train.empty else 0
                imputed_values_mar = np.full(len(X_to_impute), mode_value)
            else:
                # MAR imputation (delta=1.0)
                imputed_values_mar = safe_pmm_imputation(
                    pred_probs_recipients, pred_probs_donors, known_donors_labels,
                    k_neighbors=min(self.config.K_PMM_NEIGHBORS, len(known_donors_labels)),
                    random_state=42 + iteration
                )

            if imputed_values_mar is not None:
                df_full_mar.loc[to_impute_mask, target_ordinal] = imputed_values_mar
                logger.info(f"Imputed {len(imputed_values_mar)} MAR values.")

            # MNAR scenarios
            classes_ = [str(c) for c in model.classes_]
            for scenario, params in self.config.MNAR_SCENARIOS.items():
                delta = params['delta']
                stages = params['stages']
                
                p_adj = apply_delta_adjustment(
                    pred_probs_recipients, delta, stages, classes_
                )
                
                imputed_values_mnar = safe_pmm_imputation(
                    p_adj, pred_probs_donors, known_donors_labels,
                    k_neighbors=min(self.config.K_PMM_NEIGHBORS, len(known_donors_labels)),
                    random_state=42 + iteration
                )
                
                if imputed_values_mnar is not None:
                    all_imputed_values[scenario] = imputed_values_mnar
                    logger.info(f"Imputed {len(imputed_values_mnar)} values for MNAR scenario '{scenario}'.")

            if iteration == self.config.N_IMPUTATIONS - 1:
                self.last_model_artifacts['ordinal_model'] = model
                self.last_model_artifacts['p_recipients'] = pred_probs_recipients
                self.last_model_artifacts['p_donors'] = pred_probs_donors
                self.last_model_artifacts['y_donors_iter'] = known_donors_labels
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.Series(
                        model.feature_importances_, index=features_to_use
                    ).sort_values(ascending=False)
                    logger.debug("Top 10 important features for ordinal imputation:")
                    logger.debug(feature_importance.head(10).to_string())
        else:
            logger.info("No ordinal values to impute in this iteration.")

        return df_full_mar, all_imputed_values

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

    def _aggregate_proportions(self, imputation_results: List[Dict[str, pd.DataFrame]]) -> Dict:
        """Aggregates proportions from multiple imputation runs for each scenario."""
        scenario_proportions = {scenario: [] for scenario in self.config.MNAR_SCENARIOS.keys()}
        scenario_proportions['mar'] = []

        for run_dict in imputation_results:
            for scenario_name, df in run_dict.items():
                if scenario_name in scenario_proportions:
                    proportions = df[self.config.TARGET_VARIABLE].value_counts(normalize=True)
                    scenario_proportions[scenario_name].append(proportions)
        
        return scenario_proportions 

