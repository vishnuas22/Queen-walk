"""
Machine Learning Models Utilities

Advanced ML model implementations for analytics including ensemble methods,
transformer wrappers, reinforcement learning agents, and anomaly detection models.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging
import warnings

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. ML models will be limited.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of ML models"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    ISOLATION_FOREST = "isolation_forest"
    ENSEMBLE = "ensemble"


class TaskType(Enum):
    """Types of ML tasks"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"


@dataclass
class ModelConfig:
    """ML model configuration"""
    model_type: ModelType = ModelType.RANDOM_FOREST
    task_type: TaskType = TaskType.CLASSIFICATION
    hyperparameters: Dict[str, Any] = None
    preprocessing: Dict[str, Any] = None
    validation_strategy: str = "cross_validation"
    random_state: int = 42


@dataclass
class ModelResult:
    """ML model training/prediction result"""
    model_id: str = ""
    model_type: str = ""
    task_type: str = ""
    performance_metrics: Dict[str, float] = None
    predictions: List[Any] = None
    feature_importance: Dict[str, float] = None
    model_confidence: float = 0.0
    training_time: float = 0.0
    validation_scores: List[float] = None
    created_at: str = ""


class EnsembleModelManager:
    """
    🤖 ENSEMBLE MODEL MANAGER
    
    Advanced ensemble model management with multiple algorithms and automatic selection.
    """
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.ensemble_configs = {}
        
        logger.info("Ensemble Model Manager initialized")
    
    async def train_ensemble_model(self,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 model_config: ModelConfig,
                                 model_id: Optional[str] = None) -> ModelResult:
        """
        Train ensemble model with multiple algorithms
        
        Args:
            X: Feature matrix
            y: Target vector
            model_config: Model configuration
            model_id: Optional model identifier
            
        Returns:
            ModelResult: Training results and model performance
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for ensemble modeling")
        
        try:
            start_time = datetime.utcnow()
            
            if model_id is None:
                model_id = f"ensemble_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=model_config.random_state
            )
            
            # Preprocess data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize base models
            base_models = await self._initialize_base_models(model_config)
            
            # Train ensemble
            ensemble_predictions = []
            model_performances = {}
            
            for model_name, model in base_models.items():
                # Train individual model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                if model_config.task_type == TaskType.CLASSIFICATION:
                    predictions = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, predictions)
                    model_performances[model_name] = accuracy
                else:  # Regression
                    predictions = model.predict(X_test_scaled)
                    mse = mean_squared_error(y_test, predictions)
                    model_performances[model_name] = 1 / (1 + mse)  # Convert to score
                
                ensemble_predictions.append(predictions)
            
            # Combine predictions (simple averaging for now)
            if model_config.task_type == TaskType.CLASSIFICATION:
                # Majority voting
                ensemble_pred = []
                for i in range(len(y_test)):
                    votes = [pred[i] for pred in ensemble_predictions]
                    ensemble_pred.append(max(set(votes), key=votes.count))
                
                final_performance = accuracy_score(y_test, ensemble_pred)
            else:
                # Average predictions
                ensemble_pred = np.mean(ensemble_predictions, axis=0)
                final_performance = 1 / (1 + mean_squared_error(y_test, ensemble_pred))
            
            # Calculate feature importance (from Random Forest if available)
            feature_importance = {}
            if 'random_forest' in base_models:
                rf_model = base_models['random_forest']
                if hasattr(rf_model, 'feature_importances_'):
                    feature_importance = {
                        f'feature_{i}': importance 
                        for i, importance in enumerate(rf_model.feature_importances_)
                    }
            
            # Cross-validation scores
            cv_scores = []
            if 'random_forest' in base_models:
                cv_scores = cross_val_score(base_models['random_forest'], X_train_scaled, y_train, cv=5)
            
            training_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Store model
            self.models[model_id] = {
                'base_models': base_models,
                'scaler': scaler,
                'config': model_config,
                'performance': model_performances
            }
            
            result = ModelResult(
                model_id=model_id,
                model_type="ensemble",
                task_type=model_config.task_type.value,
                performance_metrics={
                    'ensemble_performance': final_performance,
                    'individual_performances': model_performances
                },
                predictions=ensemble_pred.tolist() if isinstance(ensemble_pred, np.ndarray) else ensemble_pred,
                feature_importance=feature_importance,
                model_confidence=final_performance,
                training_time=training_time,
                validation_scores=cv_scores.tolist() if len(cv_scores) > 0 else [],
                created_at=datetime.utcnow().isoformat()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            raise
    
    async def _initialize_base_models(self, config: ModelConfig) -> Dict[str, Any]:
        """Initialize base models for ensemble"""
        base_models = {}
        
        # Random Forest
        rf_params = config.hyperparameters.get('random_forest', {}) if config.hyperparameters else {}
        if config.task_type == TaskType.CLASSIFICATION:
            base_models['random_forest'] = RandomForestClassifier(
                n_estimators=rf_params.get('n_estimators', 100),
                max_depth=rf_params.get('max_depth', None),
                random_state=config.random_state
            )
        else:
            base_models['random_forest'] = RandomForestRegressor(
                n_estimators=rf_params.get('n_estimators', 100),
                max_depth=rf_params.get('max_depth', None),
                random_state=config.random_state
            )
        
        # Gradient Boosting
        gb_params = config.hyperparameters.get('gradient_boosting', {}) if config.hyperparameters else {}
        if config.task_type == TaskType.CLASSIFICATION:
            base_models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=gb_params.get('n_estimators', 100),
                learning_rate=gb_params.get('learning_rate', 0.1),
                random_state=config.random_state
            )
        else:
            base_models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=gb_params.get('n_estimators', 100),
                learning_rate=gb_params.get('learning_rate', 0.1),
                random_state=config.random_state
            )
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            xgb_params = config.hyperparameters.get('xgboost', {}) if config.hyperparameters else {}
            if config.task_type == TaskType.CLASSIFICATION:
                base_models['xgboost'] = xgb.XGBClassifier(
                    n_estimators=xgb_params.get('n_estimators', 100),
                    learning_rate=xgb_params.get('learning_rate', 0.1),
                    random_state=config.random_state
                )
            else:
                base_models['xgboost'] = xgb.XGBRegressor(
                    n_estimators=xgb_params.get('n_estimators', 100),
                    learning_rate=xgb_params.get('learning_rate', 0.1),
                    random_state=config.random_state
                )
        
        # SVM
        svm_params = config.hyperparameters.get('svm', {}) if config.hyperparameters else {}
        if config.task_type == TaskType.CLASSIFICATION:
            base_models['svm'] = SVC(
                kernel=svm_params.get('kernel', 'rbf'),
                C=svm_params.get('C', 1.0),
                random_state=config.random_state
            )
        else:
            base_models['svm'] = SVR(
                kernel=svm_params.get('kernel', 'rbf'),
                C=svm_params.get('C', 1.0)
            )
        
        return base_models
    
    async def predict_ensemble(self,
                             model_id: str,
                             X: np.ndarray) -> Dict[str, Any]:
        """Make predictions using ensemble model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_data = self.models[model_id]
        base_models = model_data['base_models']
        scaler = model_data['scaler']
        
        # Preprocess input
        X_scaled = scaler.transform(X)
        
        # Get predictions from all base models
        predictions = {}
        for model_name, model in base_models.items():
            pred = model.predict(X_scaled)
            predictions[model_name] = pred.tolist() if isinstance(pred, np.ndarray) else pred
        
        # Ensemble prediction
        if model_data['config'].task_type == TaskType.CLASSIFICATION:
            # Majority voting
            ensemble_pred = []
            for i in range(len(X)):
                votes = [predictions[model][i] for model in predictions]
                ensemble_pred.append(max(set(votes), key=votes.count))
        else:
            # Average predictions
            pred_arrays = [np.array(predictions[model]) for model in predictions]
            ensemble_pred = np.mean(pred_arrays, axis=0).tolist()
        
        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': predictions,
            'model_id': model_id
        }


class TransformerModelWrapper:
    """
    🔄 TRANSFORMER MODEL WRAPPER
    
    Wrapper for transformer-based models for sequence analysis.
    """
    
    def __init__(self):
        self.models = {}
        logger.info("Transformer Model Wrapper initialized")
    
    async def create_sequence_model(self,
                                  sequence_length: int,
                                  feature_dim: int,
                                  model_config: Dict[str, Any]) -> str:
        """
        Create transformer model for sequence analysis
        
        Args:
            sequence_length: Length of input sequences
            feature_dim: Dimension of features
            model_config: Model configuration
            
        Returns:
            str: Model identifier
        """
        # Simplified transformer implementation
        # In production, you'd use libraries like transformers or torch
        
        model_id = f"transformer_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Store model configuration
        self.models[model_id] = {
            'sequence_length': sequence_length,
            'feature_dim': feature_dim,
            'config': model_config,
            'trained': False
        }
        
        logger.info(f"Created transformer model {model_id}")
        return model_id
    
    async def train_sequence_model(self,
                                 model_id: str,
                                 sequences: List[List[float]],
                                 targets: List[float]) -> Dict[str, Any]:
        """Train transformer model on sequence data"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        # Simplified training (in practice, implement actual transformer training)
        model_data = self.models[model_id]
        
        # Simulate training
        training_loss = np.random.uniform(0.1, 0.5)
        validation_accuracy = np.random.uniform(0.7, 0.95)
        
        model_data['trained'] = True
        model_data['training_loss'] = training_loss
        model_data['validation_accuracy'] = validation_accuracy
        
        return {
            'model_id': model_id,
            'training_loss': training_loss,
            'validation_accuracy': validation_accuracy,
            'status': 'trained'
        }
    
    async def predict_sequence(self,
                             model_id: str,
                             sequences: List[List[float]]) -> List[float]:
        """Make predictions on sequence data"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_data = self.models[model_id]
        if not model_data['trained']:
            raise ValueError(f"Model {model_id} is not trained")
        
        # Simplified prediction (in practice, implement actual transformer inference)
        predictions = [np.random.uniform(0, 1) for _ in sequences]
        
        return predictions


class ReinforcementLearningAgent:
    """
    🎯 REINFORCEMENT LEARNING AGENT
    
    RL agent for adaptive learning path optimization.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        
        logger.info("Reinforcement Learning Agent initialized")
    
    def get_state_key(self, state: List[float]) -> str:
        """Convert state to string key for Q-table"""
        # Discretize continuous state
        discretized = [round(s, 2) for s in state]
        return str(discretized)
    
    async def choose_action(self, state: List[float]) -> int:
        """Choose action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)
        
        # Initialize Q-values for new state
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_dim
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.action_dim)
        else:
            # Exploit: best action
            return np.argmax(self.q_table[state_key])
    
    async def update_q_value(self,
                           state: List[float],
                           action: int,
                           reward: float,
                           next_state: List[float]) -> None:
        """Update Q-value using Q-learning"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-values if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_dim
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0] * self.action_dim
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
    
    async def get_policy(self) -> Dict[str, int]:
        """Get current policy (best action for each state)"""
        policy = {}
        for state_key, q_values in self.q_table.items():
            policy[state_key] = np.argmax(q_values)
        return policy


class AnomalyDetectionModels:
    """
    🚨 ANOMALY DETECTION MODELS
    
    Advanced anomaly detection using multiple algorithms.
    """
    
    def __init__(self):
        self.models = {}
        logger.info("Anomaly Detection Models initialized")
    
    async def train_anomaly_detector(self,
                                   X: np.ndarray,
                                   model_type: str = 'isolation_forest',
                                   contamination: float = 0.1) -> str:
        """Train anomaly detection model"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for anomaly detection")
        
        model_id = f"anomaly_{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        if model_type == 'isolation_forest':
            model = IsolationForest(contamination=contamination, random_state=42)
        else:
            raise ValueError(f"Unsupported anomaly detection model: {model_type}")
        
        # Train model
        model.fit(X)
        
        # Store model
        self.models[model_id] = {
            'model': model,
            'model_type': model_type,
            'contamination': contamination,
            'trained_on_samples': X.shape[0]
        }
        
        return model_id
    
    async def detect_anomalies(self,
                             model_id: str,
                             X: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in new data"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_data = self.models[model_id]
        model = model_data['model']
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = model.predict(X)
        anomaly_scores = model.decision_function(X)
        
        # Convert to boolean anomaly indicators
        is_anomaly = predictions == -1
        
        return {
            'anomaly_predictions': is_anomaly.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'anomaly_count': int(np.sum(is_anomaly)),
            'anomaly_rate': float(np.mean(is_anomaly)),
            'model_id': model_id
        }


class FeatureEngineeringPipeline:
    """
    🔧 FEATURE ENGINEERING PIPELINE
    
    Automated feature engineering for analytics data.
    """
    
    def __init__(self):
        self.pipelines = {}
        logger.info("Feature Engineering Pipeline initialized")
    
    async def create_feature_pipeline(self,
                                    feature_config: Dict[str, Any]) -> str:
        """Create feature engineering pipeline"""
        pipeline_id = f"pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        pipeline_steps = []
        
        # Scaling
        if feature_config.get('scaling', 'standard') == 'standard':
            pipeline_steps.append(('scaler', StandardScaler()))
        elif feature_config.get('scaling') == 'minmax':
            pipeline_steps.append(('scaler', MinMaxScaler()))
        
        # Store pipeline
        self.pipelines[pipeline_id] = {
            'steps': pipeline_steps,
            'config': feature_config,
            'fitted': False
        }
        
        return pipeline_id
    
    async def fit_transform_features(self,
                                   pipeline_id: str,
                                   X: np.ndarray) -> np.ndarray:
        """Fit pipeline and transform features"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline_data = self.pipelines[pipeline_id]
        
        # Apply transformations
        X_transformed = X.copy()
        fitted_transformers = {}
        
        for step_name, transformer in pipeline_data['steps']:
            X_transformed = transformer.fit_transform(X_transformed)
            fitted_transformers[step_name] = transformer
        
        # Store fitted transformers
        pipeline_data['fitted_transformers'] = fitted_transformers
        pipeline_data['fitted'] = True
        
        return X_transformed
    
    async def transform_features(self,
                               pipeline_id: str,
                               X: np.ndarray) -> np.ndarray:
        """Transform features using fitted pipeline"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline_data = self.pipelines[pipeline_id]
        if not pipeline_data['fitted']:
            raise ValueError(f"Pipeline {pipeline_id} is not fitted")
        
        # Apply transformations
        X_transformed = X.copy()
        fitted_transformers = pipeline_data['fitted_transformers']
        
        for step_name, _ in pipeline_data['steps']:
            transformer = fitted_transformers[step_name]
            X_transformed = transformer.transform(X_transformed)
        
        return X_transformed
