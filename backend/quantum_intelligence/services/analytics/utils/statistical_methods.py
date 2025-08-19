"""
Statistical Methods Utilities

Advanced statistical analysis methods for research-grade analytics including
Bayesian inference, causal inference, time series analysis, and hypothesis testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import math
import warnings

# Try to import advanced statistical libraries
try:
    import scipy.stats as stats
    from scipy import optimize
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some statistical methods will be limited.")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some ML methods will be limited.")

logger = logging.getLogger(__name__)


class StatisticalTestType(Enum):
    """Types of statistical tests"""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    CHI_SQUARE = "chi_square"
    ANOVA = "anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    CORRELATION = "correlation"
    REGRESSION = "regression"


@dataclass
class StatisticalResult:
    """Statistical analysis result"""
    test_type: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    significance_level: float = 0.05
    power: Optional[float] = None
    sample_size: Optional[int] = None


@dataclass
class BayesianResult:
    """Bayesian analysis result"""
    posterior_mean: float
    posterior_std: float
    credible_interval: Tuple[float, float]
    bayes_factor: Optional[float] = None
    prior_parameters: Dict[str, float] = None
    likelihood_parameters: Dict[str, float] = None


class StatisticalAnalyzer:
    """
    📊 STATISTICAL ANALYZER
    
    Comprehensive statistical analysis toolkit for research-grade analytics.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.analysis_history = []
        
        logger.info("Statistical Analyzer initialized")
    
    def perform_hypothesis_test(self,
                              data1: List[float],
                              data2: Optional[List[float]] = None,
                              test_type: StatisticalTestType = StatisticalTestType.T_TEST,
                              alternative: str = 'two-sided') -> StatisticalResult:
        """
        Perform hypothesis testing with effect size calculation
        
        Args:
            data1: First dataset
            data2: Second dataset (for two-sample tests)
            test_type: Type of statistical test
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
            
        Returns:
            StatisticalResult: Comprehensive test results
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for statistical testing")
        
        try:
            if test_type == StatisticalTestType.T_TEST:
                if data2 is None:
                    # One-sample t-test
                    statistic, p_value = stats.ttest_1samp(data1, 0)
                    effect_size = np.mean(data1) / np.std(data1, ddof=1)  # Cohen's d
                else:
                    # Two-sample t-test
                    statistic, p_value = stats.ttest_ind(data1, data2, alternative=alternative)
                    effect_size = self._calculate_cohens_d(data1, data2)
                
                # Calculate confidence interval
                if data2 is None:
                    ci = stats.t.interval(1 - self.significance_level, len(data1) - 1,
                                        loc=np.mean(data1), scale=stats.sem(data1))
                else:
                    pooled_se = np.sqrt(np.var(data1, ddof=1)/len(data1) + np.var(data2, ddof=1)/len(data2))
                    diff_mean = np.mean(data1) - np.mean(data2)
                    df = len(data1) + len(data2) - 2
                    ci = stats.t.interval(1 - self.significance_level, df,
                                        loc=diff_mean, scale=pooled_se)
            
            elif test_type == StatisticalTestType.MANN_WHITNEY:
                if data2 is None:
                    raise ValueError("Mann-Whitney test requires two datasets")
                statistic, p_value = stats.mannwhitneyu(data1, data2, alternative=alternative)
                effect_size = self._calculate_rank_biserial_correlation(data1, data2)
                ci = (np.nan, np.nan)  # CI not directly available for Mann-Whitney
            
            elif test_type == StatisticalTestType.WILCOXON:
                if data2 is None:
                    statistic, p_value = stats.wilcoxon(data1, alternative=alternative)
                else:
                    statistic, p_value = stats.wilcoxon(data1, data2, alternative=alternative)
                effect_size = self._calculate_wilcoxon_effect_size(data1, data2)
                ci = (np.nan, np.nan)  # CI calculation complex for Wilcoxon
            
            elif test_type == StatisticalTestType.CHI_SQUARE:
                # Assuming data1 is observed frequencies, data2 is expected frequencies
                if data2 is None:
                    raise ValueError("Chi-square test requires expected frequencies")
                statistic, p_value = stats.chisquare(data1, data2)
                effect_size = self._calculate_cramers_v(data1, data2)
                ci = (np.nan, np.nan)  # CI not standard for chi-square
            
            else:
                raise ValueError(f"Unsupported test type: {test_type}")
            
            # Interpret results
            interpretation = self._interpret_statistical_result(
                p_value, effect_size, test_type
            )
            
            # Calculate statistical power (simplified)
            power = self._estimate_statistical_power(data1, data2, effect_size, test_type)
            
            result = StatisticalResult(
                test_type=test_type.value,
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=ci,
                interpretation=interpretation,
                significance_level=self.significance_level,
                power=power,
                sample_size=len(data1) + (len(data2) if data2 else 0)
            )
            
            self.analysis_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in hypothesis testing: {e}")
            raise
    
    def _calculate_cohens_d(self, data1: List[float], data2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(data1), len(data2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(data1, ddof=1) + 
                             (n2 - 1) * np.var(data2, ddof=1)) / (n1 + n2 - 2))
        return (np.mean(data1) - np.mean(data2)) / pooled_std
    
    def _calculate_rank_biserial_correlation(self, data1: List[float], data2: List[float]) -> float:
        """Calculate rank-biserial correlation for Mann-Whitney test"""
        n1, n2 = len(data1), len(data2)
        u_statistic, _ = stats.mannwhitneyu(data1, data2)
        return 1 - (2 * u_statistic) / (n1 * n2)
    
    def _calculate_wilcoxon_effect_size(self, data1: List[float], data2: Optional[List[float]]) -> float:
        """Calculate effect size for Wilcoxon test"""
        if data2 is None:
            # One-sample Wilcoxon
            n = len(data1)
            statistic, _ = stats.wilcoxon(data1)
            return statistic / (n * (n + 1) / 4)
        else:
            # Two-sample Wilcoxon (paired)
            differences = np.array(data1) - np.array(data2)
            n = len(differences)
            statistic, _ = stats.wilcoxon(differences)
            return statistic / (n * (n + 1) / 4)
    
    def _calculate_cramers_v(self, observed: List[float], expected: List[float]) -> float:
        """Calculate Cramer's V effect size for chi-square test"""
        chi2, _ = stats.chisquare(observed, expected)
        n = sum(observed)
        k = len(observed)
        return np.sqrt(chi2 / (n * (k - 1)))
    
    def _interpret_statistical_result(self, p_value: float, effect_size: float, test_type: StatisticalTestType) -> str:
        """Interpret statistical results"""
        significance = "significant" if p_value < self.significance_level else "not significant"
        
        # Effect size interpretation (Cohen's conventions)
        if abs(effect_size) < 0.2:
            effect_magnitude = "negligible"
        elif abs(effect_size) < 0.5:
            effect_magnitude = "small"
        elif abs(effect_size) < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"
        
        return f"Result is {significance} (p={p_value:.4f}) with {effect_magnitude} effect size ({effect_size:.3f})"
    
    def _estimate_statistical_power(self, data1: List[float], data2: Optional[List[float]], 
                                  effect_size: float, test_type: StatisticalTestType) -> float:
        """Estimate statistical power (simplified calculation)"""
        # This is a simplified power calculation
        # In practice, you'd use specialized libraries like statsmodels
        n = len(data1) + (len(data2) if data2 else 0)
        
        # Approximate power calculation based on effect size and sample size
        if test_type == StatisticalTestType.T_TEST:
            # Simplified power calculation for t-test
            delta = abs(effect_size) * np.sqrt(n / 2)
            power = 1 - stats.t.cdf(stats.t.ppf(1 - self.significance_level/2, n-2), n-2, delta)
            return min(power, 1.0)
        
        # For other tests, return a rough estimate
        return min(0.8 * (abs(effect_size) + 0.1) * np.log(n) / 5, 1.0)


class BayesianInference:
    """
    🎯 BAYESIAN INFERENCE
    
    Bayesian statistical analysis for probabilistic reasoning.
    """
    
    def __init__(self):
        self.inference_history = []
        logger.info("Bayesian Inference initialized")
    
    def bayesian_t_test(self,
                       data1: List[float],
                       data2: Optional[List[float]] = None,
                       prior_mean: float = 0.0,
                       prior_precision: float = 1.0) -> BayesianResult:
        """
        Perform Bayesian t-test
        
        Args:
            data1: First dataset
            data2: Second dataset (for two-sample test)
            prior_mean: Prior mean for the effect
            prior_precision: Prior precision (inverse variance)
            
        Returns:
            BayesianResult: Bayesian analysis results
        """
        try:
            if data2 is None:
                # One-sample Bayesian t-test
                n = len(data1)
                sample_mean = np.mean(data1)
                sample_var = np.var(data1, ddof=1)
                
                # Update posterior parameters
                posterior_precision = prior_precision + n / sample_var
                posterior_mean = (prior_precision * prior_mean + n * sample_mean / sample_var) / posterior_precision
                posterior_var = 1 / posterior_precision
                
                # Credible interval
                ci_lower = posterior_mean - 1.96 * np.sqrt(posterior_var)
                ci_upper = posterior_mean + 1.96 * np.sqrt(posterior_var)
                
            else:
                # Two-sample Bayesian t-test
                n1, n2 = len(data1), len(data2)
                mean1, mean2 = np.mean(data1), np.mean(data2)
                var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
                
                # Pooled variance
                pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
                
                # Difference in means
                diff_mean = mean1 - mean2
                diff_var = pooled_var * (1/n1 + 1/n2)
                
                # Update posterior
                posterior_precision = prior_precision + 1 / diff_var
                posterior_mean = (prior_precision * prior_mean + diff_mean / diff_var) / posterior_precision
                posterior_var = 1 / posterior_precision
                
                # Credible interval
                ci_lower = posterior_mean - 1.96 * np.sqrt(posterior_var)
                ci_upper = posterior_mean + 1.96 * np.sqrt(posterior_var)
            
            # Calculate Bayes factor (simplified)
            bayes_factor = self._calculate_bayes_factor(posterior_mean, posterior_var, prior_mean, 1/prior_precision)
            
            result = BayesianResult(
                posterior_mean=posterior_mean,
                posterior_std=np.sqrt(posterior_var),
                credible_interval=(ci_lower, ci_upper),
                bayes_factor=bayes_factor,
                prior_parameters={'mean': prior_mean, 'precision': prior_precision},
                likelihood_parameters={'sample_size': len(data1) + (len(data2) if data2 else 0)}
            )
            
            self.inference_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in Bayesian t-test: {e}")
            raise
    
    def _calculate_bayes_factor(self, posterior_mean: float, posterior_var: float,
                              prior_mean: float, prior_var: float) -> float:
        """Calculate Bayes factor (simplified)"""
        # This is a simplified Bayes factor calculation
        # In practice, you'd use more sophisticated methods
        
        # Evidence for alternative hypothesis vs null hypothesis
        null_likelihood = stats.norm.pdf(0, posterior_mean, np.sqrt(posterior_var))
        alt_likelihood = stats.norm.pdf(posterior_mean, posterior_mean, np.sqrt(posterior_var))
        
        return alt_likelihood / max(null_likelihood, 1e-10)


class CausalInference:
    """
    🔗 CAUSAL INFERENCE
    
    Causal analysis methods for understanding cause-and-effect relationships.
    """
    
    def __init__(self):
        self.causal_analyses = []
        logger.info("Causal Inference initialized")
    
    def propensity_score_matching(self,
                                 treatment_data: pd.DataFrame,
                                 control_data: pd.DataFrame,
                                 covariates: List[str],
                                 outcome_variable: str) -> Dict[str, Any]:
        """
        Perform propensity score matching for causal inference
        
        Args:
            treatment_data: Data for treatment group
            control_data: Data for control group
            covariates: List of covariate column names
            outcome_variable: Name of outcome variable
            
        Returns:
            Dict: Causal inference results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for propensity score matching")
        
        try:
            # Combine data and create treatment indicator
            treatment_data = treatment_data.copy()
            control_data = control_data.copy()
            treatment_data['treatment'] = 1
            control_data['treatment'] = 0
            
            combined_data = pd.concat([treatment_data, control_data], ignore_index=True)
            
            # Estimate propensity scores (simplified logistic regression)
            from sklearn.linear_model import LogisticRegression
            
            X = combined_data[covariates]
            y = combined_data['treatment']
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit propensity score model
            ps_model = LogisticRegression()
            ps_model.fit(X_scaled, y)
            
            # Calculate propensity scores
            propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
            combined_data['propensity_score'] = propensity_scores
            
            # Perform matching (nearest neighbor)
            matched_pairs = self._nearest_neighbor_matching(combined_data)
            
            # Calculate treatment effect
            treatment_effect = self._calculate_treatment_effect(matched_pairs, outcome_variable)
            
            result = {
                'treatment_effect': treatment_effect,
                'matched_pairs': len(matched_pairs),
                'propensity_score_model': ps_model,
                'balance_statistics': self._calculate_balance_statistics(matched_pairs, covariates)
            }
            
            self.causal_analyses.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in propensity score matching: {e}")
            raise
    
    def _nearest_neighbor_matching(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform nearest neighbor matching based on propensity scores"""
        treatment_group = data[data['treatment'] == 1]
        control_group = data[data['treatment'] == 0]
        
        matched_pairs = []
        
        for _, treated_unit in treatment_group.iterrows():
            # Find nearest control unit
            distances = np.abs(control_group['propensity_score'] - treated_unit['propensity_score'])
            nearest_control_idx = distances.idxmin()
            nearest_control = control_group.loc[nearest_control_idx]
            
            matched_pairs.append({
                'treated_unit': treated_unit,
                'control_unit': nearest_control,
                'distance': distances.min()
            })
            
            # Remove matched control unit to avoid reuse
            control_group = control_group.drop(nearest_control_idx)
        
        return pd.DataFrame(matched_pairs)
    
    def _calculate_treatment_effect(self, matched_pairs: pd.DataFrame, outcome_variable: str) -> Dict[str, float]:
        """Calculate average treatment effect from matched pairs"""
        treatment_outcomes = [pair['treated_unit'][outcome_variable] for _, pair in matched_pairs.iterrows()]
        control_outcomes = [pair['control_unit'][outcome_variable] for _, pair in matched_pairs.iterrows()]
        
        ate = np.mean(treatment_outcomes) - np.mean(control_outcomes)
        ate_se = np.sqrt(np.var(treatment_outcomes) + np.var(control_outcomes)) / np.sqrt(len(matched_pairs))
        
        return {
            'average_treatment_effect': ate,
            'standard_error': ate_se,
            'confidence_interval': (ate - 1.96 * ate_se, ate + 1.96 * ate_se)
        }
    
    def _calculate_balance_statistics(self, matched_pairs: pd.DataFrame, covariates: List[str]) -> Dict[str, float]:
        """Calculate covariate balance statistics after matching"""
        balance_stats = {}
        
        for covariate in covariates:
            treatment_values = [pair['treated_unit'][covariate] for _, pair in matched_pairs.iterrows()]
            control_values = [pair['control_unit'][covariate] for _, pair in matched_pairs.iterrows()]
            
            # Standardized mean difference
            pooled_std = np.sqrt((np.var(treatment_values) + np.var(control_values)) / 2)
            smd = (np.mean(treatment_values) - np.mean(control_values)) / pooled_std
            
            balance_stats[covariate] = abs(smd)
        
        return balance_stats


class TimeSeriesAnalyzer:
    """
    📈 TIME SERIES ANALYZER
    
    Advanced time series analysis for temporal pattern detection.
    """
    
    def __init__(self):
        self.analysis_history = []
        logger.info("Time Series Analyzer initialized")
    
    def detect_change_points(self, data: List[float], method: str = 'variance') -> List[int]:
        """
        Detect change points in time series data
        
        Args:
            data: Time series data
            method: Change point detection method ('variance', 'mean')
            
        Returns:
            List[int]: Indices of detected change points
        """
        try:
            data = np.array(data)
            n = len(data)
            
            if method == 'variance':
                # Variance-based change point detection
                window_size = max(10, n // 20)
                change_points = []
                
                for i in range(window_size, n - window_size):
                    left_var = np.var(data[i-window_size:i])
                    right_var = np.var(data[i:i+window_size])
                    
                    # Test for significant variance change
                    f_stat = max(left_var, right_var) / min(left_var, right_var)
                    if f_stat > 2.0:  # Threshold for significant change
                        change_points.append(i)
                
                # Remove nearby change points
                filtered_points = []
                for cp in change_points:
                    if not filtered_points or cp - filtered_points[-1] > window_size:
                        filtered_points.append(cp)
                
                return filtered_points
            
            elif method == 'mean':
                # Mean-based change point detection using CUSUM
                cumsum = np.cumsum(data - np.mean(data))
                change_points = []
                
                # Find peaks in cumulative sum
                if SCIPY_AVAILABLE:
                    peaks, _ = find_peaks(np.abs(cumsum), height=2*np.std(cumsum))
                    change_points = peaks.tolist()
                
                return change_points
            
            else:
                raise ValueError(f"Unsupported change point detection method: {method}")
                
        except Exception as e:
            logger.error(f"Error in change point detection: {e}")
            raise
    
    def seasonal_decomposition(self, data: List[float], period: int) -> Dict[str, List[float]]:
        """
        Perform seasonal decomposition of time series
        
        Args:
            data: Time series data
            period: Seasonal period
            
        Returns:
            Dict: Decomposed components (trend, seasonal, residual)
        """
        try:
            data = np.array(data)
            n = len(data)
            
            # Simple moving average for trend
            trend = np.convolve(data, np.ones(period)/period, mode='same')
            
            # Detrended data
            detrended = data - trend
            
            # Seasonal component (average for each period position)
            seasonal = np.zeros(n)
            for i in range(period):
                period_values = detrended[i::period]
                seasonal[i::period] = np.mean(period_values)
            
            # Residual
            residual = data - trend - seasonal
            
            return {
                'trend': trend.tolist(),
                'seasonal': seasonal.tolist(),
                'residual': residual.tolist(),
                'original': data.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in seasonal decomposition: {e}")
            raise


class HypothesisTestingFramework:
    """
    🧪 HYPOTHESIS TESTING FRAMEWORK
    
    Comprehensive framework for hypothesis testing with multiple comparison correction.
    """
    
    def __init__(self):
        self.test_results = []
        logger.info("Hypothesis Testing Framework initialized")
    
    def multiple_comparison_correction(self,
                                     p_values: List[float],
                                     method: str = 'bonferroni') -> Dict[str, Any]:
        """
        Apply multiple comparison correction to p-values
        
        Args:
            p_values: List of p-values from multiple tests
            method: Correction method ('bonferroni', 'holm', 'fdr_bh')
            
        Returns:
            Dict: Corrected p-values and significance results
        """
        try:
            p_values = np.array(p_values)
            n_tests = len(p_values)
            
            if method == 'bonferroni':
                corrected_p = p_values * n_tests
                corrected_p = np.minimum(corrected_p, 1.0)
                
            elif method == 'holm':
                # Holm-Bonferroni correction
                sorted_indices = np.argsort(p_values)
                corrected_p = np.zeros_like(p_values)
                
                for i, idx in enumerate(sorted_indices):
                    corrected_p[idx] = p_values[idx] * (n_tests - i)
                
                # Ensure monotonicity
                sorted_corrected = corrected_p[sorted_indices]
                for i in range(1, n_tests):
                    if sorted_corrected[i] < sorted_corrected[i-1]:
                        sorted_corrected[i] = sorted_corrected[i-1]
                
                corrected_p[sorted_indices] = sorted_corrected
                corrected_p = np.minimum(corrected_p, 1.0)
                
            elif method == 'fdr_bh':
                # Benjamini-Hochberg FDR correction
                sorted_indices = np.argsort(p_values)
                corrected_p = np.zeros_like(p_values)
                
                for i, idx in enumerate(sorted_indices):
                    corrected_p[idx] = p_values[idx] * n_tests / (i + 1)
                
                # Ensure monotonicity (reverse)
                sorted_corrected = corrected_p[sorted_indices]
                for i in range(n_tests - 2, -1, -1):
                    if sorted_corrected[i] > sorted_corrected[i+1]:
                        sorted_corrected[i] = sorted_corrected[i+1]
                
                corrected_p[sorted_indices] = sorted_corrected
                corrected_p = np.minimum(corrected_p, 1.0)
                
            else:
                raise ValueError(f"Unsupported correction method: {method}")
            
            # Determine significance
            significant = corrected_p < 0.05
            
            return {
                'original_p_values': p_values.tolist(),
                'corrected_p_values': corrected_p.tolist(),
                'significant': significant.tolist(),
                'method': method,
                'n_tests': n_tests,
                'n_significant': np.sum(significant)
            }
            
        except Exception as e:
            logger.error(f"Error in multiple comparison correction: {e}")
            raise
