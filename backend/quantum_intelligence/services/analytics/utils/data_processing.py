"""
Data Processing Utilities

Advanced data processing utilities for analytics including preprocessing,
feature extraction, dimensionality reduction, data quality assessment,
and bias detection frameworks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging
import warnings

# Try to import advanced libraries
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA, FastICA, TruncatedSVD
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.impute import SimpleImputer, KNNImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some data processing features will be limited.")

try:
    import scipy.stats as stats
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataQualityIssue(Enum):
    """Types of data quality issues"""
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    DUPLICATES = "duplicates"
    INCONSISTENT_FORMAT = "inconsistent_format"
    BIAS = "bias"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    dataset_id: str = ""
    total_records: int = 0
    total_features: int = 0
    missing_value_percentage: float = 0.0
    duplicate_percentage: float = 0.0
    outlier_percentage: float = 0.0
    quality_score: float = 0.0
    issues_detected: List[DataQualityIssue] = None
    recommendations: List[str] = None
    assessment_timestamp: str = ""

    def __post_init__(self):
        if self.issues_detected is None:
            self.issues_detected = []
        if self.recommendations is None:
            self.recommendations = []


class DataPreprocessor:
    """
    🔧 DATA PREPROCESSOR
    
    Advanced data preprocessing for analytics with automatic handling
    of missing values, outliers, and data normalization.
    """
    
    def __init__(self):
        self.preprocessing_history = {}
        self.fitted_transformers = {}
        logger.info("Data Preprocessor initialized")
    
    def preprocess_dataset(self,
                          data: Union[pd.DataFrame, np.ndarray],
                          config: Dict[str, Any]) -> Tuple[Union[pd.DataFrame, np.ndarray], Dict[str, Any]]:
        """
        Comprehensive data preprocessing
        
        Args:
            data: Input dataset
            config: Preprocessing configuration
            
        Returns:
            Tuple of processed data and preprocessing report
        """
        try:
            preprocessing_id = f"preprocess_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Convert to DataFrame if numpy array
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
            
            original_shape = data.shape
            report = {'preprocessing_id': preprocessing_id, 'original_shape': original_shape}
            
            # Handle missing values
            if config.get('handle_missing', True):
                data, missing_report = self._handle_missing_values(data, config.get('missing_strategy', 'mean'))
                report['missing_values'] = missing_report
            
            # Handle outliers
            if config.get('handle_outliers', True):
                data, outlier_report = self._handle_outliers(data, config.get('outlier_method', 'iqr'))
                report['outliers'] = outlier_report
            
            # Normalize/scale data
            if config.get('normalize', True):
                data, scaling_report = self._normalize_data(data, config.get('scaling_method', 'standard'))
                report['scaling'] = scaling_report
            
            # Feature selection
            if config.get('feature_selection', False):
                data, feature_report = self._select_features(data, config)
                report['feature_selection'] = feature_report
            
            report['final_shape'] = data.shape
            report['preprocessing_timestamp'] = datetime.utcnow().isoformat()
            
            # Store preprocessing history
            self.preprocessing_history[preprocessing_id] = report
            
            return data, report
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise
    
    def _handle_missing_values(self, data: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values in dataset"""
        missing_before = data.isnull().sum().sum()
        missing_percentage = (missing_before / (data.shape[0] * data.shape[1])) * 100
        
        if missing_before == 0:
            return data, {'missing_count': 0, 'missing_percentage': 0.0, 'strategy': 'none'}
        
        if SKLEARN_AVAILABLE:
            if strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif strategy == 'median':
                imputer = SimpleImputer(strategy='median')
            elif strategy == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
            elif strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            else:
                imputer = SimpleImputer(strategy='mean')
            
            # Apply imputation to numeric columns only
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
                self.fitted_transformers[f'imputer_{strategy}'] = imputer
        else:
            # Simple fallback imputation
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if strategy == 'mean':
                data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
            elif strategy == 'median':
                data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
            else:
                data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
        
        missing_after = data.isnull().sum().sum()
        
        return data, {
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'missing_percentage_before': float(missing_percentage),
            'strategy': strategy
        }
    
    def _handle_outliers(self, data: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle outliers in dataset"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        original_rows = len(data)
        
        if len(numeric_columns) == 0:
            return data, {'outliers_removed': 0, 'method': method}
        
        if method == 'iqr':
            for col in numeric_columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                outliers_removed += outlier_mask.sum()
                
                # Cap outliers instead of removing
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'zscore' and SCIPY_AVAILABLE:
            for col in numeric_columns:
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                outlier_mask = z_scores > 3
                outliers_removed += outlier_mask.sum()
                
                # Cap outliers
                threshold = data[col].quantile(0.99)
                data[col] = data[col].clip(upper=threshold)
        
        return data, {
            'outliers_detected': int(outliers_removed),
            'outlier_percentage': float(outliers_removed / original_rows * 100),
            'method': method
        }
    
    def _normalize_data(self, data: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Normalize/scale numeric data"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return data, {'normalized_columns': 0, 'method': method}
        
        if SKLEARN_AVAILABLE:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
            self.fitted_transformers[f'scaler_{method}'] = scaler
        else:
            # Simple normalization fallback
            if method == 'standard':
                data[numeric_columns] = (data[numeric_columns] - data[numeric_columns].mean()) / data[numeric_columns].std()
            elif method == 'minmax':
                data[numeric_columns] = (data[numeric_columns] - data[numeric_columns].min()) / (data[numeric_columns].max() - data[numeric_columns].min())
        
        return data, {
            'normalized_columns': len(numeric_columns),
            'method': method,
            'columns_normalized': list(numeric_columns)
        }
    
    def _select_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Select most relevant features"""
        if 'target_column' not in config:
            return data, {'features_selected': data.shape[1], 'method': 'none'}
        
        target_col = config['target_column']
        if target_col not in data.columns:
            return data, {'features_selected': data.shape[1], 'method': 'none'}
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        k = config.get('k_features', min(10, X.shape[1]))
        
        if SKLEARN_AVAILABLE and len(X.columns) > 0:
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Reconstruct dataframe
            result_data = pd.DataFrame(X_selected, columns=selected_features, index=data.index)
            result_data[target_col] = y
            
            return result_data, {
                'features_selected': len(selected_features),
                'original_features': X.shape[1],
                'selected_feature_names': selected_features,
                'method': 'univariate_selection'
            }
        
        return data, {'features_selected': data.shape[1], 'method': 'none'}


class FeatureExtractor:
    """
    🎯 FEATURE EXTRACTOR
    
    Advanced feature extraction for analytics data including temporal,
    statistical, and domain-specific features.
    """
    
    def __init__(self):
        self.extraction_history = {}
        logger.info("Feature Extractor initialized")
    
    def extract_temporal_features(self, data: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
        """Extract temporal features from timestamp data"""
        if timestamp_column not in data.columns:
            return data
        
        # Convert to datetime if not already
        data[timestamp_column] = pd.to_datetime(data[timestamp_column])
        
        # Extract temporal features
        data['hour'] = data[timestamp_column].dt.hour
        data['day_of_week'] = data[timestamp_column].dt.dayofweek
        data['month'] = data[timestamp_column].dt.month
        data['quarter'] = data[timestamp_column].dt.quarter
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        
        # Time-based cyclical features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        return data
    
    def extract_statistical_features(self, data: pd.DataFrame, group_column: str = None) -> pd.DataFrame:
        """Extract statistical features from numeric columns"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return data
        
        if group_column and group_column in data.columns:
            # Group-based statistical features
            for col in numeric_columns:
                if col != group_column:
                    group_stats = data.groupby(group_column)[col].agg(['mean', 'std', 'min', 'max'])
                    data = data.merge(group_stats, left_on=group_column, right_index=True, suffixes=('', f'_{col}_group'))
        else:
            # Global statistical features
            for col in numeric_columns:
                data[f'{col}_zscore'] = (data[col] - data[col].mean()) / data[col].std()
                data[f'{col}_percentile'] = data[col].rank(pct=True)
        
        return data
    
    def extract_interaction_features(self, data: pd.DataFrame, max_interactions: int = 10) -> pd.DataFrame:
        """Extract interaction features between numeric columns"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return data
        
        interactions_created = 0
        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns[i+1:], i+1):
                if interactions_created >= max_interactions:
                    break
                
                # Multiplicative interaction
                data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
                interactions_created += 1
                
                if interactions_created >= max_interactions:
                    break
            
            if interactions_created >= max_interactions:
                break
        
        return data


class DimensionalityReducer:
    """
    📉 DIMENSIONALITY REDUCER
    
    Advanced dimensionality reduction techniques for high-dimensional analytics data.
    """
    
    def __init__(self):
        self.fitted_reducers = {}
        logger.info("Dimensionality Reducer initialized")
    
    def reduce_dimensions(self,
                         data: Union[pd.DataFrame, np.ndarray],
                         method: str = 'pca',
                         n_components: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reduce dimensionality of data
        
        Args:
            data: Input data
            method: Reduction method ('pca', 'ica', 'svd')
            n_components: Number of components to keep
            
        Returns:
            Tuple of reduced data and reduction report
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Returning original data.")
            return np.array(data), {'method': 'none', 'components': data.shape[1] if hasattr(data, 'shape') else 0}
        
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            data_array = data.select_dtypes(include=[np.number]).values
        else:
            data_array = np.array(data)
        
        if n_components is None:
            n_components = min(data_array.shape[1], data_array.shape[0] // 2)
        
        reducer_id = f"{method}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            if method == 'pca':
                reducer = PCA(n_components=n_components)
            elif method == 'ica':
                reducer = FastICA(n_components=n_components, random_state=42)
            elif method == 'svd':
                reducer = TruncatedSVD(n_components=n_components, random_state=42)
            else:
                reducer = PCA(n_components=n_components)
            
            reduced_data = reducer.fit_transform(data_array)
            
            # Store fitted reducer
            self.fitted_reducers[reducer_id] = reducer
            
            # Calculate explained variance if available
            explained_variance = None
            if hasattr(reducer, 'explained_variance_ratio_'):
                explained_variance = reducer.explained_variance_ratio_.tolist()
                total_variance = sum(explained_variance)
            else:
                total_variance = None
            
            report = {
                'method': method,
                'original_dimensions': data_array.shape[1],
                'reduced_dimensions': n_components,
                'explained_variance_ratio': explained_variance,
                'total_variance_explained': total_variance,
                'reducer_id': reducer_id
            }
            
            return reduced_data, report
            
        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {e}")
            return data_array, {'method': 'error', 'error': str(e)}


class DataQualityAssessment:
    """
    🔍 DATA QUALITY ASSESSMENT
    
    Comprehensive data quality assessment and reporting.
    """
    
    def __init__(self):
        self.assessment_history = {}
        logger.info("Data Quality Assessment initialized")
    
    def assess_data_quality(self, data: pd.DataFrame, dataset_id: str = None) -> DataQualityReport:
        """
        Comprehensive data quality assessment
        
        Args:
            data: Dataset to assess
            dataset_id: Optional dataset identifier
            
        Returns:
            DataQualityReport: Comprehensive quality report
        """
        if dataset_id is None:
            dataset_id = f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Basic statistics
        total_records = len(data)
        total_features = len(data.columns)
        
        # Missing values assessment
        missing_values = data.isnull().sum().sum()
        missing_percentage = (missing_values / (total_records * total_features)) * 100
        
        # Duplicate assessment
        duplicates = data.duplicated().sum()
        duplicate_percentage = (duplicates / total_records) * 100
        
        # Outlier assessment (for numeric columns)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        
        if len(numeric_columns) > 0 and SCIPY_AVAILABLE:
            for col in numeric_columns:
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                outlier_count += (z_scores > 3).sum()
        
        outlier_percentage = (outlier_count / (total_records * len(numeric_columns))) * 100 if len(numeric_columns) > 0 else 0
        
        # Identify issues
        issues = []
        recommendations = []
        
        if missing_percentage > 5:
            issues.append(DataQualityIssue.MISSING_VALUES)
            recommendations.append(f"Address {missing_percentage:.1f}% missing values using imputation")
        
        if duplicate_percentage > 1:
            issues.append(DataQualityIssue.DUPLICATES)
            recommendations.append(f"Remove {duplicates} duplicate records")
        
        if outlier_percentage > 5:
            issues.append(DataQualityIssue.OUTLIERS)
            recommendations.append(f"Handle {outlier_percentage:.1f}% outliers using capping or transformation")
        
        if total_records < 100:
            issues.append(DataQualityIssue.INSUFFICIENT_DATA)
            recommendations.append("Consider collecting more data for robust analysis")
        
        # Calculate overall quality score
        quality_score = max(0, 100 - missing_percentage - duplicate_percentage - outlier_percentage/2)
        quality_score = min(100, quality_score)
        
        report = DataQualityReport(
            dataset_id=dataset_id,
            total_records=total_records,
            total_features=total_features,
            missing_value_percentage=missing_percentage,
            duplicate_percentage=duplicate_percentage,
            outlier_percentage=outlier_percentage,
            quality_score=quality_score,
            issues_detected=issues,
            recommendations=recommendations,
            assessment_timestamp=datetime.utcnow().isoformat()
        )
        
        # Store assessment
        self.assessment_history[dataset_id] = report
        
        return report


class BiasDetectionFramework:
    """
    ⚖️ BIAS DETECTION FRAMEWORK
    
    Framework for detecting various types of bias in analytics data and models.
    """
    
    def __init__(self):
        self.bias_reports = {}
        logger.info("Bias Detection Framework initialized")
    
    def detect_sampling_bias(self, data: pd.DataFrame, population_stats: Dict[str, Any] = None) -> Dict[str, Any]:
        """Detect sampling bias in dataset"""
        bias_report = {
            'bias_type': 'sampling_bias',
            'detected': False,
            'severity': 'low',
            'details': {},
            'recommendations': []
        }
        
        # Check for obvious sampling issues
        if len(data) < 30:
            bias_report['detected'] = True
            bias_report['severity'] = 'high'
            bias_report['details']['small_sample'] = True
            bias_report['recommendations'].append("Increase sample size for more representative data")
        
        # Check for missing demographic representation (if available)
        if population_stats:
            for feature, expected_dist in population_stats.items():
                if feature in data.columns:
                    actual_dist = data[feature].value_counts(normalize=True).to_dict()
                    
                    # Compare distributions (simplified)
                    for category, expected_prop in expected_dist.items():
                        actual_prop = actual_dist.get(category, 0)
                        if abs(actual_prop - expected_prop) > 0.1:  # 10% threshold
                            bias_report['detected'] = True
                            bias_report['details'][f'{feature}_bias'] = {
                                'expected': expected_prop,
                                'actual': actual_prop,
                                'difference': abs(actual_prop - expected_prop)
                            }
        
        return bias_report
    
    def detect_measurement_bias(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect measurement bias in data collection"""
        bias_report = {
            'bias_type': 'measurement_bias',
            'detected': False,
            'severity': 'low',
            'details': {},
            'recommendations': []
        }
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        # Check for suspicious patterns
        for col in numeric_columns:
            # Check for excessive rounding (many values ending in 0 or 5)
            if len(data[col].dropna()) > 0:
                rounded_values = data[col].dropna() % 1 == 0
                if rounded_values.mean() > 0.8:  # 80% rounded
                    bias_report['detected'] = True
                    bias_report['details'][f'{col}_rounding'] = rounded_values.mean()
                    bias_report['recommendations'].append(f"Check measurement precision for {col}")
                
                # Check for digit preference (ending in 0 or 5)
                last_digits = (data[col].dropna() % 10).astype(int)
                digit_counts = last_digits.value_counts()
                if len(digit_counts) > 0:
                    max_digit_freq = digit_counts.max() / len(last_digits)
                    if max_digit_freq > 0.3:  # 30% of values end in same digit
                        bias_report['detected'] = True
                        bias_report['details'][f'{col}_digit_preference'] = max_digit_freq
        
        return bias_report
    
    def detect_confirmation_bias(self, data: pd.DataFrame, hypothesis: str = None) -> Dict[str, Any]:
        """Detect potential confirmation bias in data analysis"""
        bias_report = {
            'bias_type': 'confirmation_bias',
            'detected': False,
            'severity': 'low',
            'details': {},
            'recommendations': [
                "Use pre-registered analysis plans",
                "Apply multiple testing corrections",
                "Consider alternative hypotheses"
            ]
        }
        
        # This is a simplified check - in practice, would need more context
        # Check for suspicious perfect correlations or results
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) >= 2:
            correlation_matrix = data[numeric_columns].corr()
            
            # Check for suspiciously high correlations
            high_corr_count = 0
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = abs(correlation_matrix.iloc[i, j])
                    if corr_value > 0.95 and corr_value < 1.0:  # Very high but not perfect
                        high_corr_count += 1
            
            if high_corr_count > len(numeric_columns) // 2:
                bias_report['detected'] = True
                bias_report['details']['high_correlations'] = high_corr_count
                bias_report['recommendations'].append("Verify high correlations are not due to data manipulation")
        
        return bias_report
    
    def comprehensive_bias_assessment(self, data: pd.DataFrame, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive bias assessment across multiple dimensions"""
        assessment_id = f"bias_assessment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Run all bias detection methods
        sampling_bias = self.detect_sampling_bias(data, context.get('population_stats') if context else None)
        measurement_bias = self.detect_measurement_bias(data)
        confirmation_bias = self.detect_confirmation_bias(data, context.get('hypothesis') if context else None)
        
        # Aggregate results
        all_biases = [sampling_bias, measurement_bias, confirmation_bias]
        detected_biases = [bias for bias in all_biases if bias['detected']]
        
        overall_report = {
            'assessment_id': assessment_id,
            'total_biases_detected': len(detected_biases),
            'bias_details': {
                'sampling_bias': sampling_bias,
                'measurement_bias': measurement_bias,
                'confirmation_bias': confirmation_bias
            },
            'overall_bias_risk': 'high' if len(detected_biases) >= 2 else 'medium' if len(detected_biases) == 1 else 'low',
            'recommendations': list(set([rec for bias in all_biases for rec in bias['recommendations']])),
            'assessment_timestamp': datetime.utcnow().isoformat()
        }
        
        # Store assessment
        self.bias_reports[assessment_id] = overall_report
        
        return overall_report
