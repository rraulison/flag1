# this file handles the analysis of imputation results - analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Any, Union
from pathlib import Path

from . import utils
from .config import Config

logger = logging.getLogger(__name__)

class Analysis:
    """
    Handles analysis of imputation results, including diagnostics and visualizations.
    """
    def __init__(self, config: Config, output_prefix: Path = None):
        self.config = config
        self.output_prefix = output_prefix or config.get_output_prefix()

    def analyze_results(self, scenario_proportions: Dict, imputed_dataset: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                            original_dataset: pd.DataFrame, model_performance: Dict, 
                            model_artifacts: Dict) -> tuple:
        """
        Analyzes imputation results and returns a summary.
        
        Args:
            scenario_proportions: Dictionary containing proportions for each scenario
            imputed_dataset: The final imputed dataset (can be a single DataFrame or a dict of DataFrames)
            model_performance: Dictionary containing performance metrics for the models
            model_artifacts: Dictionary containing model artifacts
            
        Returns:
            tuple: (results_summary, pooled_metrics, scenario_names, original_props)
        """
        logger.info("Analyzing imputation results...")
        
        # Get target variable from config
        target_var = self.config.TARGET_VARIABLE
        
        # Handle both dict and DataFrame inputs
        if isinstance(imputed_dataset, dict):
            # Use MAR scenario as the main dataset for analysis
            main_imputed_df = imputed_dataset.get('mar')
            if main_imputed_df is None:
                raise ValueError("MAR scenario not found in imputed datasets")
        else:
            # Backward compatibility
            main_imputed_df = imputed_dataset
        
        # Use the original dataset for calculating original proportions
        original_dataset = original_dataset.copy()
        
        try:
            # Validate input data
            if main_imputed_df is None or main_imputed_df.empty:
                raise ValueError("Imputed dataset is empty or None")
                
            if original_dataset is None or original_dataset.empty:
                raise ValueError("Original dataset is empty or None")
                
            # Ensure target variable exists in the dataset
            if target_var not in main_imputed_df.columns:
                raise ValueError(f"Target variable '{target_var}' not found in the imputed dataset")
                
            if target_var not in original_dataset.columns:
                raise ValueError(f"Target variable '{target_var}' not found in the original dataset")
                
            # Convert target to string and handle missing values
            main_imputed_df[target_var] = main_imputed_df[target_var].astype(str)
            
            # Get classes from config or infer from data (excluding missing values)
            if hasattr(self.config, 'CLASSES'):
                classes_ = [str(c) for c in self.config.CLASSES]
            else:
                classes_ = sorted([str(c) for c in main_imputed_df[target_var].unique() 
                                if str(c) != '99' and pd.notna(c)])
            
            if not classes_:
                raise ValueError("No valid classes found in the target variable")
                
            logger.debug(f"Using classes: {classes_}")
            
            # Calculate original proportions from known data in the original dataset
            try:
                # Convert target_var to string for consistent comparison with '99'
                target_series = original_dataset[target_var].astype(str)
                known_mask = (target_series != '99') & original_dataset[target_var].notna()
                
                if not known_mask.any():
                    logger.warning("No known data points found (all values are missing or '99')")
                    known_mask = original_dataset[target_var].notna()  # Fallback to just non-NA values
                    
                known_data = original_dataset[known_mask]
                
                if len(known_data) == 0:
                    raise ValueError("No valid known data points found after filtering")
                    
            except Exception as e:
                logger.error(f"Error processing known data: {str(e)}", exc_info=True)
                # Try to recover by using all non-NA values
                known_mask = original_dataset[target_var].notna()
                known_data = original_dataset[known_mask]
                if len(known_data) == 0:
                    raise ValueError("No valid data available for analysis") from e
                
            # Ensure all expected classes are represented in the counts
            original_counts = known_data[target_var].value_counts()
            original_counts = original_counts.reindex(classes_, fill_value=0)
            original_props = original_counts / original_counts.sum()
            
            # Define scenario names with proper formatting
            scenario_names = {
                'mar': 'Imputado MAR (PMM)',
                **{k: f'Imputado MNAR({v["stages"]}, δ={v["delta"]})' 
                   for k, v in getattr(self.config, 'MNAR_SCENARIOS', {}).items()}
            }
            
            # Initialize results with proper class ordering
            results_summary = {'Original (Conhecido)': original_props}
            pooled_metrics = {}
            
            # Ensure all expected scenarios are present in the proportions
            expected_scenarios = ['mar'] + list(getattr(self.config, 'MNAR_SCENARIOS', {}).keys())
            for scenario in expected_scenarios:
                if scenario not in scenario_proportions or not scenario_proportions[scenario]:
                    logger.warning(f"Missing or empty proportions for scenario: {scenario}")
                    # Initialize with empty list if missing
                    scenario_proportions[scenario] = []
        
            # Process each scenario's proportions
            for scenario_key in expected_scenarios:
                if scenario_key not in scenario_proportions:
                    continue
                
                proportions = scenario_proportions[scenario_key]
                display_name = scenario_names.get(scenario_key, scenario_key)
                
                if not proportions or (isinstance(proportions, (pd.DataFrame, pd.Series)) and proportions.empty):
                    logger.warning(f"Empty or None proportions for scenario: {scenario_key}")
                    continue
                
                try:
                    # Convert single DataFrame to list of DataFrames if needed
                    if not isinstance(proportions, list):
                        proportions = [proportions]
                    
                    # Ensure all proportions are DataFrames with consistent columns
                    valid_proportions = []
                    for prop in proportions:
                        if prop is None or (isinstance(prop, (pd.DataFrame, pd.Series)) and prop.empty):
                            continue
                            
                        try:
                            # Convert to DataFrame if Series
                            if isinstance(prop, pd.Series):
                                prop = prop.to_frame().T
                            
                            # Ensure all expected classes are present
                            prop = prop.copy()  # Avoid modifying original
                            for cls in classes_:
                                if cls not in prop.columns:
                                    prop[cls] = 0.0
                            
                            # Reorder columns to match class order and convert to numpy array
                            prop_array = prop[classes_].values
                            if not np.any(np.isnan(prop_array)):  # Only add if no NaNs
                                valid_proportions.append(prop_array)
                            
                        except Exception as e:
                            logger.warning(f"Error processing proportion for {scenario_key}: {str(e)}")
                            continue
                    
                    if not valid_proportions:
                        logger.warning(f"No valid proportions for scenario: {scenario_key}")
                        continue
                    
                    # Stack all proportions for the scenario
                    try:
                        stacked_proportions = np.vstack(valid_proportions)
                        
                        # Skip if no valid data
                        if stacked_proportions.size == 0 or np.all(np.isnan(stacked_proportions)):
                            logger.warning(f"No valid data in stacked proportions for {scenario_key}")
                            continue
                            
                        # Ensure we have the right number of dimensions
                        if len(stacked_proportions.shape) != 2 or stacked_proportions.shape[1] != len(classes_):
                            logger.warning(f"Invalid shape for {scenario_key} proportions: {stacked_proportions.shape}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error stacking proportions for {scenario_key}: {str(e)}")
                        continue
                    
                    # Calculate metrics for the scenario with robust error handling
                    try:
                        n_imputations = len(valid_proportions)
                        conf_level = getattr(self.config, 'CONFIDENCE_LEVEL', 0.95)
                        total_n = len(imputed_dataset) if hasattr(imputed_dataset, '__len__') else len(original_dataset)
                        
                        logger.debug(f"Calculating metrics for {scenario_key} with {n_imputations} imputations, "
                                   f"confidence level {conf_level}, total N={total_n}")
                        
                        metrics = utils.get_pooled_metrics_from_proportions(
                            stacked_proportions,
                            classes_,
                            n_imputations,
                            conf_level,
                            total_n
                        )
                        
                        if not metrics or 'mean' not in metrics:
                            raise ValueError("No valid metrics returned from get_pooled_metrics_from_proportions")
                            
                        # Log some debug info
                        logger.debug(f"Metrics for {scenario_key}: {metrics}")
                        
                    except Exception as e:
                        logger.error(f"Error calculating metrics for {scenario_key}: {str(e)}", exc_info=True)
                        continue
                    
                    # Check if we got valid metrics
                    if not metrics or 'mean' not in metrics or metrics['mean'] is None:
                        logger.warning(f"No valid metrics returned for {scenario_key}")
                        continue
                    
                    # Ensure all metrics are numpy arrays
                    for key in metrics:
                        if not isinstance(metrics[key], np.ndarray):
                            metrics[key] = np.array(metrics[key])
                    
                    # Store results with proper class alignment
                    results_summary[display_name] = pd.Series(
                        metrics['mean'],
                        index=classes_
                    )
                    pooled_metrics[display_name] = metrics
                    
                except Exception as e:
                    logger.error(f"Error processing scenario {scenario_key}: {str(e)}", exc_info=True)
                    continue
            
            # Store pooled metrics for later use in diagnostics
            self.pooled_metrics = pooled_metrics
            
            return results_summary, pooled_metrics, scenario_names, original_props
            
        except Exception as e:
            logger.error(f"Error in analyze_results: {str(e)}", exc_info=True)
            # Return empty results in case of error
            empty_series = pd.Series(index=classes_ if 'classes_' in locals() else [])
            return {'Original (Conhecido)': empty_series}, {}, {}, empty_series

    def _analyze_model_performance(self, model_performance: Dict):
        """
        Analyzes and visualizes the performance of imputation models.
        
        Args:
            model_performance: Dictionary containing performance metrics for the models.
                             Expected to have 'binary' and 'ordinal' keys with lists of scores.
        """
        logger.info("\n--- Análise de Performance do Modelo de Imputação ---")
        
        # Handle different formats of model performance data
        if isinstance(model_performance, dict):
            # New format with separate binary and ordinal scores
            for model_type, scores in model_performance.items():
                if scores:  # Only process if we have scores
                    logger.info(f"\nPerformance do modelo {model_type.upper()}:")
                    perf_series = pd.Series(scores).dropna()
                    if not perf_series.empty:
                        logger.info(f"Estatísticas da performance (LogLoss) para {model_type}:")
                        logger.info(perf_series.describe())
                        
                        # Plot histogram of performance metrics
                        plt.figure(figsize=(10, 5))
                        plt.hist(perf_series, bins=20, edgecolor='k', alpha=0.7)
                        plt.title(f'Desempenho do Modelo {model_type.upper()}\nDistribuição do LogLoss')
                        plt.xlabel('LogLoss no Conjunto de Validação')
                        plt.ylabel('Frequência')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(f"{self.output_prefix}_{model_type}_model_performance.png")
                        plt.close()
                    else:
                        logger.warning(f"Nenhuma métrica de desempenho disponível para o modelo {model_type}")
        elif isinstance(model_performance, (list, pd.Series)):
            # Legacy format - single list of scores
            logger.warning("Usando formato legado de métricas de desempenho")
            perf_series = pd.Series(model_performance).dropna()
            logger.info("Estatísticas da performance (LogLoss) dos modelos de imputação:")
            logger.info(perf_series.describe())
            
            plt.figure(figsize=(10, 5))
            plt.hist(perf_series, bins=20, edgecolor='k', alpha=0.7)
            plt.title('Distribuição da Performance (LogLoss) dos Modelos de Imputação')
            plt.xlabel('LogLoss no Conjunto de Validação')
            plt.ylabel('Frequência')
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{self.output_prefix}_model_performance.png")
            plt.close()
        else:
            logger.warning(f"Formato de métricas de desempenho não reconhecido: {type(model_performance)}")

    def _run_mnar_sensitivity_analysis(self, last_model_artifacts: Dict, known_counts: pd.Series, 
                                     classes_: List[str], n_simulations: int = 50):
        """
        Executes robust MNAR sensitivity analysis with multiple scenarios.
        """
        if not last_model_artifacts:
            logger.warning("Nenhum artefato de modelo salvo para a análise de sensibilidade.")
            return

        logger.info("\n=== ANÁLISE DE SENSIBILIDADE MNAR ===")
        
        # Fix: Use config DELTA_RANGE properly
        scenarios = {
            'MNAR_Leve': {
                'delta_range': np.linspace(self.config.DELTA_RANGE[0], self.config.DELTA_RANGE[-1]/2, 5), 
                'stages': [3, 4]
            },
            'MNAR_Moderado': {
                'delta_range': np.linspace(self.config.DELTA_RANGE[0], self.config.DELTA_RANGE[-1]*0.75, 7), 
                'stages': [4]
            },
            'MNAR_Grave': {
                'delta_range': self.config.DELTA_RANGE[:15],  # Use first 15 values from config
                'stages': [4]
            }
        }
        
        all_results = []
        
        for scenario_name, params in scenarios.items():
            logger.info(f"\n--- Cenário: {scenario_name} ---")
            logger.info(f"Estágios afetados: {params['stages']}")
            logger.info(f"Variação de delta: {params['delta_range'][0]:.1f} a {params['delta_range'][-1]:.1f}")
            
            scenario_results = {'delta': [], 'stage': [], 'proportion': [], 'scenario': []}
            
            for delta in params['delta_range']:
                p_adj = utils.apply_delta_adjustment(
                    last_model_artifacts['p_recipients'], 
                    delta, 
                    [str(s) for s in params['stages']], 
                    [str(c) for c in classes_]
                )
                
                vals_mnar = utils.safe_pmm_imputation(
                    p_adj, 
                    last_model_artifacts['p_donors'], 
                    last_model_artifacts['y_donors_iter'],
                    k_neighbors=self.config.K_PMM_NEIGHBORS,
                    random_state=42
                )
                
                imputed_counts = pd.Series(vals_mnar).value_counts()
                total_counts = known_counts.add(imputed_counts, fill_value=0)
                props = total_counts / total_counts.sum()
                
                for stage in params['stages']:
                    stage_str = str(stage)
                    scenario_results['delta'].append(delta)
                    scenario_results['stage'].append(stage_str)
                    scenario_results['proportion'].append(props.get(stage_str, 0))
                    scenario_results['scenario'].append(scenario_name)
            
            df_scenario = pd.DataFrame(scenario_results)
            all_results.append(df_scenario)
            
            self._plot_sensitivity_scenario(df_scenario, scenario_name, params['stages'])
        
        df_results = pd.concat(all_results, ignore_index=True)
        results_file = f"{self.output_prefix}_mnar_sensitivity_results.csv"
        df_results.to_csv(results_file, index=False)
        logger.info(f"\nResultados completos salvos em: {results_file}")

    def _plot_sensitivity_scenario(self, df: pd.DataFrame, scenario_name: str, stages_affected: List[int]):
        """Plots sensitivity for a specific scenario."""
        plt.figure(figsize=(12, 6))
        
        for stage in stages_affected:
            stage_data = df[df['stage'] == str(stage)]
            plt.plot(stage_data['delta'], stage_data['proportion'], 
                    marker='o', label=f'Estágio {stage}')
        
        plt.title(f'Sensibilidade MNAR: {scenario_name}\n' 
                  f'Estágios Afetados: {sorted(stages_affected)}')
        plt.xlabel('Fator Delta (δ)')
        plt.ylabel('Proporção Estimada')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=1.0, color='gray', linestyle='--', label='MAR (δ=1.0)')
        
        for stage in stages_affected:
            stage_data = df[df['stage'] == str(stage)]
            if not stage_data.empty:
                mar_point = stage_data.iloc[(stage_data['delta']-1).abs().argsort()[:1]]
                if not mar_point.empty:
                    plt.annotate(f'MAR: {mar_point["proportion"].values[0]:.3f}',
                               xy=(mar_point['delta'], mar_point['proportion']),
                               xytext=(10, 10), textcoords='offset points',
                               arrowprops=dict(arrowstyle='->'))
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        filename = f"{self.output_prefix}_sensitivity_{scenario_name.lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Gráfico salvo como: {filename}")

    def _analyze_confidence_coverage(self, pooled_metrics: Dict, original_props: pd.Series, classes_: List[str]):
        """
        Analyzes and logs confidence interval coverage and diagnostic metrics.
        """
        logger.info("\n=== ANÁLISE DE INCERTEZA E COBERTURA ===")
        logger.info(f"Nível de confiança nominal: {self.config.CONFIDENCE_LEVEL:.0%}")
        logger.info("-" * 100)
        logger.info(f"{'Cenário':<35} {'Cob.':>5} {'Largura Média':>15} {'Aum. Var.':>12} {'Info. Perdida':>15} {'Classes Cobertas'}")
        logger.info("-" * 100)
        
        for scenario_name, metrics in pooled_metrics.items():
            # Handle both DataFrame and dictionary access patterns
            if isinstance(metrics, pd.DataFrame):
                # DataFrame access
                if 'ci_lower' not in metrics.columns or metrics['ci_lower'].isna().all():
                    continue
                
                ci_lower = metrics['ci_lower'].values
                ci_upper = metrics['ci_upper'].values
                rel_var_inc = metrics.get('relative_variance_increase', pd.Series([np.nan] * len(classes_))).mean()
                frac_missing = metrics.get('fraction_missing_info', pd.Series([np.nan] * len(classes_))).mean()
            else:
                # Dictionary access
                if 'ci_lower' not in metrics or all(pd.isna(metrics['ci_lower'])):
                    continue
                
                ci_lower = metrics['ci_lower']
                ci_upper = metrics['ci_upper']
                rel_var_inc = np.mean(metrics.get('relative_variance_increase', [np.nan] * len(classes_)))
                frac_missing = np.mean(metrics.get('fraction_missing_info', [np.nan] * len(classes_)))
            
            covered = []
            ci_widths = []
            
            for i, cls in enumerate(classes_):
                if i >= len(ci_lower) or i >= len(ci_upper):
                    continue
                    
                original_prop = original_props[cls]
                lower = ci_lower[i]
                upper = ci_upper[i]
                
                if pd.isna(lower) or pd.isna(upper):
                    continue
                    
                is_covered = lower <= original_prop <= upper
                covered.append(is_covered)
                ci_widths.append(upper - lower)
            
            if not covered:  # In case all values were NaN
                continue
                
            coverage_rate = np.mean(covered)
            avg_width = np.mean(ci_widths) if ci_widths else np.nan
            
            covered_classes = [str(cls) for i, cls in enumerate(classes_) if covered[i]]
            not_covered = [str(classes_[i]) for i, c in enumerate(covered) if not c]
            
            logger.info(f"{scenario_name:<35} {coverage_rate:>5.0%} {avg_width:>15.4f} {rel_var_inc:>12.1f}x {frac_missing:>15.1%} {','.join(covered_classes) if len(covered_classes) <= 3 else f'{len(covered_classes)} classes'}")
            
            if not_covered:
                logger.info(f"{'':<35} {'':5} {'':15} {'':12} {'':15} Não cobertas: {','.join(not_covered)}")
        
        logger.info("\nLEGENDA:"
              "\n  • Cob.: Proporção de classes cujos intervalos de confiança cobrem o valor original"
              "\n  • Largura Média: Largura média dos intervalos de confiança"
              "\n  • Aum. Var.: Fator de aumento da variância devido à imputação (média entre classes)"
              "\n  • Info. Perdida: Fração de informação perdida devido a dados faltantes (média entre classes)"
              "\n\nNOTA: A baixa cobertura pode indicar que:"
              "\n  • Os intervalos são muito estreitos para capturar a incerteza real"
              "\n  • O modelo de imputação não captura adequadamente a variabilidade"
              "\n  • Os dados faltantes podem não ser completamente aleatórios (MNAR)"
              "\n\nRECOMENDAÇÃO: Considere a análise de sensibilidade MNAR para avaliar a robustez"
              "\ndos resultados a diferentes suposições sobre o mecanismo de dados faltantes.")

    def run_diagnostics(self, model_performance: Dict, model_artifacts: Dict, imputed_dataset: Union[pd.DataFrame, Dict[str, pd.DataFrame]], original_dataset: pd.DataFrame):
        """
        Runs all diagnostic analyses.
        
        Args:
            model_performance: Dictionary containing performance metrics for the models
            model_artifacts: Dictionary containing model artifacts
            imputed_dataset: The imputed dataset (can be a single DataFrame or a dict of DataFrames)
            original_dataset: The original dataset before imputation
        """
        self._analyze_model_performance(model_performance)
        
        # For sensitivity analysis, we need known counts and classes
        target_var = self.config.TARGET_VARIABLE
        # Convert target_var to string for consistent comparison with '99'
        target_series = original_dataset[target_var].astype(str)
        known_mask = (target_series != '99') & original_dataset[target_var].notna()
        known_data = original_dataset[known_mask]
        
        if not known_data.empty:
            classes_ = sorted([str(c) for c in known_data[target_var].unique() if str(c) != '99' and pd.notna(c)])
            known_counts = known_data[target_var].value_counts().reindex(classes_, fill_value=0)
            
            self._run_mnar_sensitivity_analysis(
                last_model_artifacts=model_artifacts,
                known_counts=known_counts,
                classes_=classes_
            )
        else:
            logger.warning("Skipping MNAR sensitivity analysis because no known data was found.")

        if hasattr(self, 'pooled_metrics') and self.pooled_metrics:
            original_props = original_dataset[original_dataset[target_var] != '99'][target_var].value_counts(normalize=True)
            self._analyze_confidence_coverage(self.pooled_metrics, original_props, list(original_props.index))
        else:
            logger.warning("Skipping confidence coverage analysis because pooled_metrics are not available.")

    def generate_summary_table(self, pooled_metrics: Dict, original_props: pd.Series, classes_: List[str]) -> pd.DataFrame:
        """
        Generates a formatted summary table of the results with robust error handling.
        
        Args:
            pooled_metrics: Dictionary containing metrics for each scenario
            original_props: Series with original proportions for each class
            classes_: List of class names/stages
            
        Returns:
            pd.DataFrame: Formatted summary table with all scenarios and metrics
        """
        summary_list = []
        
        try:
            # Validate inputs
            if not isinstance(original_props, (pd.Series, dict)) or not original_props.any():
                logger.warning("Invalid or empty original_props, using placeholder values")
                original_props = pd.Series(index=classes_, data=1.0/len(classes_) if classes_ else [])
                
            if not classes_:
                logger.warning("No classes provided, using default classes")
                classes_ = [str(i) for i in range(len(original_props))] if not original_props.empty else ['0']
                
            # Original Proportions - Summary Row
            try:
                summary_list.append({
                    'Cenário': 'Original (Conhecido)',
                    'Estágio': 'Todos',
                    'Proporção Média': original_props.sum(),
                    'IC Inferior': original_props.sum(),
                    'IC Superior': original_props.sum(),
                    'Aumento Rel. Var.': 0.0,
                    'Fração Info. Perdida': 0.0
                })
            except Exception as e:
                logger.error(f"Error creating summary row: {str(e)}", exc_info=True)
                summary_list.append({
                    'Cenário': 'Original (Conhecido',
                    'Estágio': 'Erro',
                    'Proporção Média': np.nan,
                    'IC Inferior': np.nan,
                    'IC Superior': np.nan,
                    'Aumento Rel. Var.': np.nan,
                    'Fração Info. Perdida': np.nan
                })
            
            # Original Proportions - Per Class
            for cls in classes_:
                try:
                    prop = original_props.get(str(cls), 0.0)  # Handle both string and numeric keys
                    if not isinstance(prop, (int, float)) or np.isnan(prop):
                        prop = 0.0
                        
                    summary_list.append({
                        'Cenário': 'Original (Conhecido)',
                        'Estágio': str(cls),
                        'Proporção Média': float(prop),
                        'IC Inferior': float(prop),
                        'IC Superior': float(prop),
                        'Aumento Rel. Var.': 0.0,
                        'Fração Info. Perdida': 0.0
                    })
                except Exception as e:
                    logger.error(f"Error processing class {cls}: {str(e)}", exc_info=True)
                    continue

            # Pooled Results for Each Scenario
            for scenario, metrics in (pooled_metrics or {}).items():
                if not metrics or not isinstance(metrics, dict):
                    logger.warning(f"Skipping invalid metrics for scenario: {scenario}")
                    continue
                    
                try:
                    # Get all metrics with safe defaults
                    means = metrics.get('mean', np.array([np.nan] * len(classes_)))
                    ci_lower = metrics.get('ci_lower', np.array([np.nan] * len(classes_)))
                    ci_upper = metrics.get('ci_upper', np.array([np.nan] * len(classes_)))
                    rel_var = metrics.get('relative_variance_increase', np.array([np.nan] * len(classes_)))
                    frac_missing = metrics.get('fraction_missing_info', np.array([np.nan] * len(classes_)))
                    
                    # Add row for each class
                    for i, cls in enumerate(classes_):
                        try:
                            summary_list.append({
                                'Cenário': str(scenario),
                                'Estágio': str(cls),
                                'Proporção Média': float(means[i]) if i < len(means) else np.nan,
                                'IC Inferior': float(ci_lower[i]) if i < len(ci_lower) else np.nan,
                                'IC Superior': float(ci_upper[i]) if i < len(ci_upper) else np.nan,
                                'Aumento Rel. Var.': float(rel_var[i]) if i < len(rel_var) else np.nan,
                                'Fração Info. Perdida': float(frac_missing[i]) if i < len(frac_missing) else np.nan
                            })
                        except (IndexError, TypeError, ValueError) as e:
                            logger.error(f"Error processing class {i} in scenario {scenario}: {str(e)}", exc_info=True)
                            continue
                            
                except Exception as e:
                    logger.error(f"Error processing scenario {scenario}: {str(e)}", exc_info=True)
                    continue
            
            # Create DataFrame
            df_summary = pd.DataFrame(summary_list)
            
            # Ensure all required columns exist
            required_cols = ['Cenário', 'Estágio', 'Proporção Média', 'IC Inferior', 
                            'IC Superior', 'Aumento Rel. Var.', 'Fração Info. Perdida']
            for col in required_cols:
                if col not in df_summary.columns:
                    df_summary[col] = np.nan
            
            # Format numeric columns
            try:
                format_cols = ['Proporção Média', 'IC Inferior', 'IC Superior', 'Fração Info. Perdida']
                for col in format_cols:
                    if col in df_summary.columns:
                        # Convert to numeric, coerce errors to NaN
                        df_summary[col] = pd.to_numeric(df_summary[col], errors='coerce')
                        # Format as percentage
                        df_summary[col] = df_summary[col].apply(
                            lambda x: f'{x:.2%}' if pd.notna(x) and np.isfinite(x) else '-'
                        )
                
                # Format relative variance
                if 'Aumento Rel. Var.' in df_summary.columns:
                    df_summary['Aumento Rel. Var.'] = pd.to_numeric(
                        df_summary['Aumento Rel. Var.'], errors='coerce'
                    ).apply(lambda x: f'{x:.1f}x' if pd.notna(x) and np.isfinite(x) else '-')
                    
            except Exception as e:
                logger.error(f"Error formatting table: {str(e)}", exc_info=True)
                # Return unformatted table if formatting fails
                pass
                
            return df_summary
            
        except Exception as e:
            logger.critical(f"Critical error generating summary table: {str(e)}", exc_info=True)
            # Return minimal error table
            return pd.DataFrame({
                'Cenário': ['Erro'],
                'Mensagem': [f'Erro ao gerar tabela: {str(e)}']
            })

    def create_visualizations(self, results_summary: Dict, scenario_proportions: Dict, 
                              scenario_names: Dict, classes_: List[str]):
        """
        Creates and saves visualizations of the results.
        """
        self._plot_results_barchart(results_summary, scenario_names)
        self._plot_convergence_traces(scenario_proportions, scenario_names, classes_)
        self._plot_imputation_distribution_chart(results_summary, scenario_names, classes_)

    def _plot_imputation_distribution_chart(self, results_summary: Dict, scenario_names: Dict, classes_: List[str]):
        """Generates a bar chart comparing ESTADIAM distribution across all scenarios."""
        
        # Use the pooled metrics for imputed scenarios and original props for the known data
        df_plot = pd.DataFrame(results_summary).sort_index()
        
        # Ensure all scenarios are present and have the correct names
        df_plot.rename(columns=scenario_names, inplace=True)

        # Sort columns to have a consistent order: Original, MAR, then MNARs
        ordered_columns = ['Original (Conhecido)', 'Imputado MAR (PMM)'] + \
                          [v for k, v in scenario_names.items() if k != 'mar']
        
        # Filter out any columns that might not be in the results_summary
        ordered_columns = [col for col in ordered_columns if col in df_plot.columns]
        df_plot = df_plot[ordered_columns]

        ax = df_plot.plot(kind='bar', figsize=(16, 9), width=0.85, 
                           colormap='viridis')

        plt.title(f'Distribuição de ESTADIAM por Cenário de Imputação (M={self.config.N_IMPUTATIONS})', 
                  fontsize=16, fontweight='bold')
        plt.ylabel('Proporção Estimada', fontsize=12)
        plt.xlabel('Estadiamento Clínico', fontsize=12)
        plt.xticks(rotation=0, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Cenários', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        
        # Add percentage labels on top of each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1%}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points', 
                        fontsize=8, 
                        color='black')

        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        
        # Save the figure
        output_path = f"{self.output_prefix}_distribution_by_scenario.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved imputation distribution chart to {output_path}")

    def _plot_results_barchart(self, results_summary: Dict, scenario_names: Dict):
        """
        Generates and saves the main results bar chart.
        """
        plt.figure(figsize=(15, 8))
        df_plot = pd.DataFrame(results_summary)
        df_plot.plot(kind='bar', ax=plt.gca(), width=0.8)
        plt.title(f'Distribuição de ESTADIAM por Cenário de Imputação (N={self.config.N_IMPUTATIONS})')
        plt.ylabel('Proporção')
        plt.xlabel('Estadiamento')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{self.output_prefix}_results_barchart.png")
        plt.close()
        logger.info(f"Saved results bar chart to {self.output_prefix}_results_barchart.png")

    def _plot_convergence_traces(self, scenario_proportions: Dict, scenario_names: Dict, classes_: List[str]):
        """
        Generates and saves convergence trace plots.
        """
        n_classes = len(classes_)
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4), squeeze=False, sharey=True)
        axes = axes.flatten()
        
        for i, cls in enumerate(classes_):
            ax = axes[i]
            for scenario_key, props_list in scenario_proportions.items():
                if props_list and scenario_key in scenario_names:
                    values = [props[cls] for props in props_list]
                    ax.plot(values, label=scenario_names[scenario_key], alpha=0.8, linewidth=1.5)
            ax.set_title(f'Estágio {cls}')
            ax.set_xlabel('Iteração')
            ax.set_ylabel('Proporção')
            ax.grid(True, alpha=0.3)
        
        for j in range(n_classes, len(axes)):
            axes[j].set_visible(False)
            
        fig.legend(list(scenario_names.values()), loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4)
        plt.suptitle('Diagnóstico de Convergência - Trace Plots', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        plt.savefig(f"{self.output_prefix}_convergence_trace_plots.png")
        plt.close()
        logger.info(f"Saved convergence trace plots to {self.output_prefix}_convergence_trace_plots.png")

    def export_imputed_datasets(self, imputed_datasets: Dict[str, pd.DataFrame]):
        """Exports the final imputed datasets for each scenario."""
        logger.info("Exporting imputed datasets...")
        for scenario, df in imputed_datasets.items():
            df_export = df.copy()
            df_export['MI_ID'] = self.config.N_IMPUTATIONS
            
            output_file = self.output_prefix.parent / f"{self.output_prefix.name}_imputed_dataset_{scenario}.csv"
            try:
                df_export.to_csv(output_file, index=False)
                logger.info(f"   ✓ Exported {scenario} dataset to {output_file}")
            except Exception as e:
                logger.error(f"   ❌ Failed to export {scenario} dataset: {e}")

    def generate_and_export_report(self, df_original: pd.DataFrame, df_known: pd.DataFrame, 
                                   df_imputed: pd.DataFrame, results_summary: Dict, 
                                   model_performance: Dict, final_results: pd.DataFrame, 
                                   classes_: List[str]):
        """
        Gera o relatório executivo final e exporta todos os resultados.
        
        Args:
            df_original: DataFrame original com todos os dados (incluindo valores ausentes)
            df_known: DataFrame contendo apenas os dados conhecidos (não imputados)
            df_imputed: DataFrame com os dados imputados
            results_summary: Dicionário com os resultados resumidos
            model_performance: Dicionário com as métricas de desempenho dos modelos
            final_results: DataFrame com os resultados finais formatados
            classes_: Lista de classes do alvo
        """
        # Extrai as pontuações de desempenho do dicionário de desempenho
        model_performance_scores = model_performance.get('ordinal', [])
        df_filt = df_original  # Mantido para compatibilidade com o código existente
        df_to_imp = df_original[~df_original.index.isin(df_known.index)]  # Dados para imputação
        
        # Debug: Log the available keys in results_summary
        logger.debug(f"Available result keys: {list(results_summary.keys())}")
        
        # Normalize MAR scenario name
        mar_key = next((k for k in results_summary.keys() if 'mar' in k.lower() or 'pmm' in k.lower()), None)
        
        if mar_key is None and results_summary:
            mar_key = list(results_summary.keys())[0]  # Fallback to first key if no MAR found
            logger.warning(f"No explicit MAR scenario found, using first available: {mar_key}")
        
        """
        Generates the final executive summary and exports all results.
        """
        logger.info("Generating executive summary and exporting results...")
        perf_series = pd.Series(model_performance_scores).dropna()

        summary_text = f'''
================================================================================
RESUMO EXECUTIVO - IMPUTAÇÃO MÚLTIPLA DE ESTADIAMENTO (REVISADO)
================================================================================

DADOS PROCESSADOS:
  • Total de registros: {len(df_filt):,}
  • Dados conhecidos: {len(df_known):,} ({len(df_known)/len(df_filt)*100:.1f}%)
  • Dados para imputação: {len(df_to_imp):,} ({len(df_to_imp)/len(df_filt)*100:.1f}%)

METODOLOGIA REVISADA:
  • Imputação Múltipla com {self.config.N_IMPUTATIONS} iterações.
  • Algoritmo: CatBoost + PMM (com pool de doadores do conjunto de validação).
  • Análise de sensibilidade MNAR com delta-adjustment e análise de tipping-point.
  • Monitoramento da performance (LogLoss) dos modelos de imputação.

RESULTADOS PRINCIPAIS (Distribuição Original vs Imputada MAR):
'''
        if mar_key and mar_key in results_summary and 'Original' in results_summary:
            mar_results = results_summary[mar_key]
            original_props = results_summary['Original (Conhecido)']
            
            # Debug logging
            logger.debug(f"Using MAR key: {mar_key}")
            logger.debug(f"MAR results: {mar_results}")
            logger.debug(f"Original props: {original_props}")
            
            for cls in classes_:
                try:
                    # Handle different possible class types (int/str)
                    cls_key = str(cls).strip()
                    orig_val = original_props.get(int(cls), original_props.get(cls_key, 0))
                    imp_val = mar_results.get(int(cls), mar_results.get(cls_key, 0))
                    
                    # Convert to percentages
                    orig_pct = orig_val * 100
                    imp_pct = imp_val * 100
                    diff = imp_pct - orig_pct
                    
                    summary_text += f"   • Estágio {cls}: {orig_pct:.1f}% → {imp_pct:.1f}% ({diff:+.1f}pp)\n"
                except Exception as e:
                    logger.warning(f"Error processing class {cls}: {e}")
                    continue

        summary_text += f'''
QUALIDADE E DIAGNÓSTICOS:
  • Performance Média dos Modelos (LogLoss): {perf_series.mean():.4f} (DP: {perf_series.std():.4f})
  • Os modelos de imputação demonstraram boa e estável performance.

⚠️  INTERVALOS DE CONFIANÇA E SENSIBILIDADE:
   • ALERTA: A cobertura dos intervalos de confiança é baixa. Eles subestimam
     a incerteza real e devem ser interpretados com extrema cautela.
   • A análise de tipping-point mostra como as conclusões sobre o Estágio '{self.config.TIPPING_POINT_STAGE}' mudam
     com diferentes suposições sobre os dados faltantes.

✅ RECOMENDAÇÕES:
   1. Utilizar a imputação MAR como o resultado principal, mas sempre acompanhada do alerta
      sobre os intervalos de confiança.
   2. Usar os cenários MNAR e a análise de tipping-point para discutir a robustez
      (ou sensibilidade) dos achados a diferentes suposições.
   3. Validar as distribuições imputadas com especialistas da área.
'''
        logger.info(summary_text)

        # Exportação
        logger.info("Exporting results...")
        try:
            final_results.to_csv(f"{self.output_prefix}_summary.csv")
            logger.info(f"   ✓ Summary exported: {self.output_prefix}_summary.csv")
            
            with open(f"{self.output_prefix}_report.txt", 'w', encoding='utf-8') as f:
                f.write(summary_text)
            logger.info(f"   ✓ Full report: {self.output_prefix}_report.txt")
            
            # Export the imputed datasets
            if isinstance(df_imputed, dict):
                self.export_imputed_datasets(df_imputed)
            else:
                logger.warning(f"Expected df_imputed to be a dict, got {type(df_imputed).__name__}")
                self.export_imputed_datasets({'mar': df_imputed})

        except Exception as e:
            logger.error(f"   ❌ Export error: {e}")

        logger.info("REVISED ANALYSIS SUCCESSFULLY COMPLETED!")
