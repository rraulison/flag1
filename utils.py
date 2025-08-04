"""
Enhanced utility functions with better error handling and performance.
"""
import numpy as np
import pandas as pd
from cuml.accel import install
install()
import cuml.neighbors
from typing import List, Optional, Union, Tuple, Dict
import logging
from contextlib import contextmanager
import gc

try:
    from scipy.stats import t
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    t = None
    logging.warning("SciPy not available. Using normal approximation for CIs.")

@contextmanager
def memory_efficient_context():
    """Context manager for memory-efficient operations."""
    try:
        yield
    finally:
        gc.collect()

class PooledMetrics:
    """Class for handling pooled metrics calculations."""
    
    @staticmethod
    def calculate_rubin_metrics(props_array: np.ndarray, total_n: int, conf_level: float) -> Dict:
        """
        Enhanced Rubin's rules for multiple imputation with better handling of edge cases.
        
        Args:
            props_array: 2D array of shape (n_imputations, n_classes) with proportions
            total_n: Total number of observations
            conf_level: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Dictionary with pooled metrics including means, CIs, and variance components
        """
        n_imps, n_classes = props_array.shape
        means = np.mean(props_array, axis=0)
        
        # Handle single imputation case
        if n_imps == 1:
            return PooledMetrics._single_imputation_metrics(means, total_n, conf_level)
        
        # Rubin's rules with better numerical stability
        # Within-imputation variance (average of individual variances)
        W = np.mean([p * (1 - p) / (total_n - 1 + 1e-10) for p in props_array], axis=0)
        
        # Between-imputation variance with Bessel's correction
        B = np.var(props_array, axis=0, ddof=1)
        
        # Total variance with finite-population correction (Barnard & Rubin, 1999)
        T = W + B * (1 + 1/n_imps)
        
        # CORRECTED: Proper degrees of freedom calculation (Barnard & Rubin, 1999)
        lambda_hat = (B + B/n_imps) / (T + 1e-10)  # Fraction of missing information
        v_old = (n_imps - 1) / (lambda_hat**2 + 1e-10)  # Old degrees of freedom
        v_obs = (1 - lambda_hat) * (total_n - 1)  # Observed data DF (large sample approx)

        # Barnard-Rubin adjustment for degrees of freedom
        with np.errstate(divide='ignore', invalid='ignore'):
            df = (1 / v_old + 1 / v_obs)**-1
        
        # Calculate confidence intervals with proper t-distribution
        if SCIPY_AVAILABLE:
            alpha = 1 - conf_level
            # Ensure df is finite and positive
            df = np.nan_to_num(df, nan=1e6, posinf=1e6, neginf=1)
            df = np.clip(df, 1, 1e6)  # Clip to reasonable range
            t_critical = np.array([t.ppf(1 - alpha/2, df_val) for df_val in df])
        else:
            t_critical = 1.96  # Fallback to normal approximation
        
        # Calculate margin of error and confidence intervals
        margin = t_critical * np.sqrt(T)
        ci_lower = np.clip(means - margin, 0, 1)
        ci_upper = np.clip(means + margin, 0, 1)
        
        # Calculate relative variance increase and fraction of missing information
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_variance_increase = (B * (1 + 1/n_imps)) / (W + 1e-10)
            fraction_missing_info = (B + B/n_imps) / (T + 1e-10)
        
        # Handle edge cases
        relative_variance_increase = np.nan_to_num(relative_variance_increase, nan=0, posinf=1e6, neginf=0)
        fraction_missing_info = np.clip(fraction_missing_info, 0, 1)
        
        return {
            'mean': means,
            'std': np.sqrt(T),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'relative_variance_increase': relative_variance_increase,
            'fraction_missing_info': fraction_missing_info,
            'df': df  # Return degrees of freedom for reference
        }
    
    @staticmethod
    def _single_imputation_metrics(means: np.ndarray, total_n: int, conf_level: float) -> Dict:
        """Calculate metrics for single imputation."""
        std_errs = np.sqrt(means * (1 - means) / total_n)
        margin = 1.96 * std_errs
        
        return {
            'mean': means,
            'std': std_errs,
            'ci_lower': np.clip(means - margin, 0, 1),
            'ci_upper': np.clip(means + margin, 0, 1),
            'relative_variance_increase': np.full_like(means, np.nan),
            'fraction_missing_info': np.full_like(means, np.nan)
        }

def get_pooled_metrics_from_proportions(props_list: List[pd.Series], classes: List[str], 
                                      n_imps: int, conf_level: float, total_n: int) -> pd.DataFrame:
    """Enhanced pooled metrics calculation with better error handling."""
    if not props_list:
        logging.warning("Empty proportions list provided")
        return pd.DataFrame(index=classes)
    
    try:
        props_array = np.array([prop.reindex(classes, fill_value=0).values for prop in props_list])
        metrics = PooledMetrics.calculate_rubin_metrics(props_array, total_n, conf_level)
        return pd.DataFrame(metrics, index=classes)
    except Exception as e:
        logging.error(f"Error calculating pooled metrics: {e}")
        return pd.DataFrame(index=classes)
        
def safe_pmm_imputation(probs_recipients: np.ndarray, probs_donors: np.ndarray, 
                       y_donors: Union[pd.Series, np.ndarray], k_neighbors: int = 5, 
                       random_state: Optional[int] = None) -> np.ndarray:
    """
    Enhanced PMM with better error handling, validation, and shape checking.
    
    Args:
        probs_recipients: 2D array of shape (n_recipients, n_classes) with probability scores
        probs_donors: 2D array of shape (n_donors, n_classes) with probability scores
        y_donors: Series or array of shape (n_donors,) with class labels
        k_neighbors: Number of neighbors to consider
        random_state: Random seed for reproducibility
        
    Returns:
        Array of imputed class labels for recipients
    """
    # Input validation
    if len(probs_donors) == 0:
        logging.warning("No donors available for PMM")
        # Handle both Series and array cases
        if hasattr(y_donors, 'iloc') and len(y_donors) > 0:
            return np.full(len(probs_recipients), y_donors.iloc[0])
        elif len(y_donors) > 0:
            return np.full(len(probs_recipients), y_donors[0])
        else:
            return np.full(len(probs_recipients), '0')
    
    if len(probs_recipients) == 0:
        return np.array([])
        
    if probs_recipients.ndim != 2 or probs_donors.ndim != 2:
        raise ValueError(f"probs_recipients and probs_donors must be 2D arrays. "
                        f"Got shapes {probs_recipients.shape} and {probs_donors.shape}")
    
    if probs_recipients.shape[1] != probs_donors.shape[1]:
        raise ValueError(f"Dimensionality mismatch: probs_recipients has {probs_recipients.shape[1]} "
                        f"features but probs_donors has {probs_donors.shape[1]}")
    
    if len(probs_donors) != len(y_donors):
        raise ValueError(f"Length mismatch: probs_donors has {len(probs_donors)} samples "
                        f"but y_donors has {len(y_donors)}")
    
    rng = np.random.RandomState(random_state)
    k_eff = min(len(probs_donors), k_neighbors)
    
    try:
        # Initialize and fit nearest neighbors
        nn = cuml.neighbors.NearestNeighbors(n_neighbors=k_eff, metric='euclidean')
        nn.fit(probs_donors)
        
        # Find nearest neighbors for each recipient
        distances, neighbor_indices = nn.kneighbors(probs_recipients)
        
        # Randomly select one neighbor for each recipient
        chosen_indices = rng.randint(0, k_eff, size=len(probs_recipients))
        donor_indices = neighbor_indices[np.arange(len(probs_recipients)), chosen_indices]
        
        # Handle both pandas Series and numpy array cases
        if hasattr(y_donors, 'iloc'):
            # pandas Series case
            return y_donors.iloc[donor_indices].values
        else:
            # numpy array case
            return y_donors[donor_indices]
        
    except Exception as e:
        logging.error(f"PMM imputation failed: {e}")
        logging.error(f"Shapes - probs_recipients: {probs_recipients.shape}, "
                     f"probs_donors: {probs_donors.shape}, y_donors: {len(y_donors)}")
        
        # Fallback to random sampling - handle both cases
        if hasattr(y_donors, 'values'):
            return rng.choice(y_donors.values, size=len(probs_recipients))
        else:
            return rng.choice(y_donors, size=len(probs_recipients))

def apply_delta_adjustment_optimized(probs: np.ndarray, delta: float, target_stages: List[str], 
                                   classes: List[str]) -> np.ndarray:
    """
    VERSÃO OTIMIZADA: Delta adjustment com melhor performance.
    
    Principais otimizações:
    1. Elimina cópias desnecessárias
    2. Usa operações vetorizadas
    3. Pré-computa índices
    4. Evita reshapes desnecessários
    """
    if delta <= 0:
        logging.warning(f"Invalid delta value: {delta}. Using delta=1.0")
        delta = 1.0
    
    # Otimização 1: Pré-computar índices usando set lookup (O(1) vs O(n))
    target_set = set(target_stages)
    target_indices = np.array([i for i, cls in enumerate(classes) if cls in target_set], dtype=np.int32)
    
    if len(target_indices) == 0:
        logging.warning(f"No target stages found in classes: {target_stages}")
        return probs  # Retorna original sem cópia
    
    # Otimização 2: Aplicar delta diretamente sem cópia inicial
    probs_adj = probs.copy()  # Uma única cópia necessária
    
    # Otimização 3: Operação vetorizada mais eficiente
    probs_adj[:, target_indices] *= delta
    
    # Otimização 4: Normalização otimizada
    row_sums = probs_adj.sum(axis=1, keepdims=True)
    
    # Otimização 5: Usar np.where para evitar indexação booleana complexa
    # Isso é mais rápido que mask + indexação separada
    probs_adj = np.where(
        row_sums > 0,
        probs_adj / row_sums,
        1.0 / len(classes)  # Distribuição uniforme para linhas zeradas
    )
    
    return probs_adj

def apply_delta_adjustment_ultra_fast(probs: np.ndarray, delta: float, target_stages: List[str], 
                                    classes: List[str]) -> np.ndarray:
    """
    VERSÃO ULTRA-RÁPIDA: Para casos onde performance é crítica.
    
    Usa operações in-place quando possível e elimina validações custosas.
    """
    if delta == 1.0:
        return probs  # Nenhuma modificação necessária
    
    # Pré-computar índices uma única vez
    target_indices = np.array([i for i, cls in enumerate(classes) if cls in target_stages], dtype=np.int32)
    
    if len(target_indices) == 0:
        return probs
    
    # Trabalhar com uma cópia
    result = probs.copy()
    
    # Aplicar delta (vectorizado)
    result[:, target_indices] *= delta
    
    # Normalização ultra-rápida
    row_sums = result.sum(axis=1, keepdims=True)
    np.divide(result, row_sums, out=result, where=row_sums > 0)
    
    # Corrigir linhas com soma zero (casos raros)
    zero_mask = (row_sums == 0).ravel()
    if np.any(zero_mask):
        result[zero_mask] = 1.0 / len(classes)
    
    return result

# Classe para cache de índices (para múltiplas chamadas)
class DeltaAdjustmentCache:
    """Cache para índices de target_stages para evitar recomputação."""
    
    def __init__(self):
        self._cache = {}
    
    def get_target_indices(self, target_stages: List[str], classes: List[str]) -> np.ndarray:
        """Retorna índices cached ou computa novos."""
        key = (tuple(target_stages), tuple(classes))
        
        if key not in self._cache:
            target_set = set(target_stages)
            self._cache[key] = np.array([i for i, cls in enumerate(classes) if cls in target_set], dtype=np.int32)
        
        return self._cache[key]
    
    def apply_delta_adjustment_cached(self, probs: np.ndarray, delta: float, 
                                    target_stages: List[str], classes: List[str]) -> np.ndarray:
        """Delta adjustment com cache de índices."""
        if delta == 1.0:
            return probs
        
        target_indices = self.get_target_indices(target_stages, classes)
        
        if len(target_indices) == 0:
            return probs
        
        result = probs.copy()
        result[:, target_indices] *= delta
        
        row_sums = result.sum(axis=1, keepdims=True)
        np.divide(result, row_sums, out=result, where=row_sums > 0)
        
        zero_mask = (row_sums == 0).ravel()
        if np.any(zero_mask):
            result[zero_mask] = 1.0 / len(classes)
        
        return result

# Instância global do cache
_delta_cache = DeltaAdjustmentCache()

# Função wrapper para usar a versão otimizada
def apply_delta_adjustment(probs: np.ndarray, delta: float, target_stages: List[str], 
                         classes: List[str]) -> np.ndarray:
    """
    Função principal com seleção automática da melhor implementação.
    """
    # Para datasets pequenos, usar versão com validação completa
    if probs.shape[0] < 1000:
        return apply_delta_adjustment_optimized(probs, delta, target_stages, classes)
    
    # Para datasets grandes, usar versão ultra-rápida com cache
    return _delta_cache.apply_delta_adjustment_cached(probs, delta, target_stages, classes)

def predict_proba_in_batches(model, X: pd.DataFrame, batch_size: int = 50000, 
                           dtype: np.dtype = np.float16) -> np.ndarray:
    """
    Memory-efficient batch prediction with pre-allocation and proper cleanup.
    
    Args:
        model: Trained model with predict_proba method
        X: Input features as DataFrame
        batch_size: Number of samples per batch
        dtype: Data type for probability outputs
        
    Returns:
        np.ndarray: 2D array of predicted probabilities (n_samples, n_classes)
    """
    n_samples = X.shape[0]
    
    # Pre-allocate output array to avoid memory fragmentation
    n_classes = len(model.classes_)
    result = np.empty((n_samples, n_classes), dtype=dtype)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_X = X.iloc[start_idx:end_idx]
        
        try:
            with memory_efficient_context():
                # Get predictions for this batch
                batch_probs = model.predict_proba(batch_X).astype(dtype)
                result[start_idx:end_idx] = batch_probs
                
                # Log progress for large datasets
                if n_samples > 10 * batch_size and start_idx % (10 * batch_size) == 0:
                    logging.debug(f"Processed {end_idx}/{n_samples} samples ({end_idx/n_samples:.1%})")
                    
        except Exception as e:
            logging.error(f"Error in batch prediction [{start_idx}:{end_idx}]: {e}")
            # Fallback to uniform probabilities
            result[start_idx:end_idx] = 1.0 / n_classes
            
        # Explicitly clean up
        del batch_X
        gc.collect()
    
    return result
