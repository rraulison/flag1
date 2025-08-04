import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
import warnings
#import cudf.pandas
#cudf.pandas.install()

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data loading, preprocessing, and preparation for imputation."""
    def __init__(self, config):
        self.config = config

    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, List[str], List[int]]:
        """Loads, preprocesses, and returns the main DataFrame and feature info."""
        df_filt = self._load_data()
        df_filt = self._select_features(df_filt)
        df_filt = self._create_split_variables(df_filt)
        self._log_missing_data_info(df_filt)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module='pandas')
            df_filt = self._convert_data_types(df_filt)
        
        feature_cols = [c for c in df_filt.columns if c != self.config.TARGET_VARIABLE]
        # Re-determine cat_idx AFTER type conversions
        cat_idx = [i for i, c in enumerate(feature_cols) if df_filt[c].dtype.name in ['category', 'object']]
        
        # Apply fillna("NaN") only to actual categorical/object columns
        for col_idx in cat_idx:
            col_name = feature_cols[col_idx]
            # Ensure it's not already numeric (e.g., if it was converted from object to numeric)
            if df_filt[col_name].dtype.kind not in ['i', 'f', 'b']:
                df_filt[col_name] = df_filt[col_name].astype(str).fillna("NaN")
                logger.info(f"Corrected categorical column to handle NaN as string: {col_name}")
            
        return df_filt, feature_cols, cat_idx

    def _load_data(self) -> pd.DataFrame:
        """Loads the initial dataset."""
        logger.info(f"Loading data from {self.config.DATA_FILE}")
        return pd.read_parquet(self.config.DATA_FILE)

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selects features based on importance."""
        logger.info(f"Loading feature importance from {self.config.FEATURE_IMPORTANCE_FILE}")
        feature_importance_df = pd.read_csv(self.config.FEATURE_IMPORTANCE_FILE)
        selected_features = feature_importance_df.feature.tolist()
        selected_features.append(self.config.TARGET_VARIABLE)
        logger.info(f"Selected {len(selected_features)} features.")
        return df[selected_features]

    def _create_split_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates the 'flag_nao_estadiavel' and 'estadiamento_ordinal' variables.
        
        For flag_nao_estadiavel:
        - 1 if ESTADIAM is '88' (não estadiável)
        - 0 if ESTADIAM is one of '0'-'4' (estadiável)
        - NaN if ESTADIAM is '99' (desconhecido)
        
        For estadiamento_ordinal:
        - 0-4 for valid staging values
        - NaN where ESTADIAM is '88' or '99'
        """
        # Create flag_nao_estadiavel according to specification:
        # 1 if ESTADIAM is '88' (não estadiável)
        # 0 if ESTADIAM is one of '0'-'4'
        # NaN if ESTADIAM is '99' (desconhecido)
        df['flag_nao_estadiavel'] = np.where(
            df[self.config.TARGET_VARIABLE] == '88', 1,
            np.where(
                df[self.config.TARGET_VARIABLE].isin(['0', '1', '2', '3', '4']), 0, np.nan
            )
        )
        
        # Create estadiamento_ordinal with values 0-4 where ESTADIAM is 0-4, NaN otherwise
        ordinal_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
        df[self.config.ORDINAL_TARGET_VARIABLE] = df[self.config.TARGET_VARIABLE].map(ordinal_mapping)
        
        # Set estadiamento_ordinal to NaN where ESTADIAM is '88' or '99'
        df.loc[df[self.config.TARGET_VARIABLE].isin(['88', '99']), self.config.ORDINAL_TARGET_VARIABLE] = np.nan
        
        # Log the distribution of the new variables
        logger.info(f"Distribution of flag_nao_estadiavel:\n{df['flag_nao_estadiavel'].value_counts(dropna=False)}")
        logger.info(f"Distribution of estadiamento_ordinal:\n{df[self.config.ORDINAL_TARGET_VARIABLE].value_counts(dropna=False).sort_index()}")
        
        return df

    def _log_missing_data_info(self, df: pd.DataFrame):
        """Logs information about missing data."""
        na_cols = df.columns[df.isna().any()].tolist()
        if na_cols:
            logger.info(f"Columns with missing data: {na_cols}")
            for col in na_cols:
                logger.info(f"  {col}: {df[col].isna().sum()} missing values")
        else:
            logger.info("No columns with missing data.")

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts data types for compatibility."""
        # Create a copy to avoid chained assignment warnings
        df_converted = df.copy()

        for col in df_converted.columns:
            if col == self.config.TARGET_VARIABLE:
                continue

            # Handle object columns first
            if df_converted[col].dtype.name == 'object':
                # Try to convert to datetime
                s_dt = pd.to_datetime(df_converted[col], errors='coerce')
                if s_dt.notna().sum() > 0.5 * df_converted[col].notna().sum() and s_dt.notna().sum() > 0:
                    logger.info(f"Column '{col}' identified as date and converted to float64.")
                    s_numeric = s_dt.view('int64')
                    s_float = s_numeric.astype('float64')
                    s_float[s_numeric == pd.NaT.value] = np.nan
                    df_converted[col] = s_float
                # If not date, don't convert to numeric automatically
                # Just keep as is

            # DON'T convert category columns to numeric
            # If it's period dtype, convert to timestamp numeric
            elif pd.api.types.is_period_dtype(df_converted[col]):
                logger.info(f"Column '{col}' is period dtype and will be converted to timestamp (int64).");
                df_converted[col] = df_converted[col].dt.to_timestamp().astype('int64') // 10**9

        return df_converted

    def prepare_data_for_imputation(self, df_filt: pd.DataFrame, feature_cols: List[str]):
        """Splits data into known and to-be-imputed sets."""
        target = self.config.ORDINAL_TARGET_VARIABLE  # Use the new ordinal variable
        
        # Known data now includes cases where estadiamento_ordinal is not null
        df_known = df_filt[df_filt[target].notna()].copy().reset_index(drop=True)
        
        # To-impute data are cases where estadiamento_ordinal is null AND flag_nao_estadiavel is 0
        df_to_imp = df_filt[
            (df_filt[target].isna()) & (df_filt['flag_nao_estadiavel'] == 0)
        ].copy().reset_index(drop=True)

        X_known = df_known[feature_cols].copy()
        y_known = df_known[target].copy()
        X_to_imp = df_to_imp[feature_cols].copy()
        
        classes_ = [str(int(c)) for c in sorted(y_known.unique())]
        known_counts = y_known.value_counts()
        
        logger.info(f"Data prepared for imputation: "
                    f"{len(X_known)} known, {len(X_to_imp)} to impute.")
        
        return X_known, y_known, X_to_imp, classes_, known_counts, df_known, df_to_imp

