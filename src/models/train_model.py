"""
Model Training Module
Comprehensive ML model training pipeline for CSK match prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# Setup logging with dd/mm/yyyy format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)

class CSKModelTrainer:
    """Comprehensive model training for CSK match prediction"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_name = ""
        
        # Initialize model configurations
        self._setup_models()
    
    def _setup_models(self):
        """Setup model configurations"""
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 50, 100]
                }
            },
            'extra_trees': {
                'model': ExtraTreesClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
            }
        }
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, 
                        test_size: float = 0.2) -> Dict[str, Any]:
        """Train all models and return results"""
        logger.info("🚀 Starting comprehensive model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train each model
        for model_name, config in self.model_configs.items():
            logger.info(f"🔄 Training {model_name}...")
            
            try:
                # Create pipeline with scaling
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', config['model'])
                ])
                
                # Hyperparameter tuning
                param_grid = {f'classifier__{k}': v for k, v in config['params'].items()}
                
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                # Fit model
                grid_search.fit(X_train, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                
                # Evaluate model
                results = self._evaluate_model(
                    best_model, X_train, X_test, y_train, y_test, model_name
                )
                
                # Store results
                self.models[model_name] = best_model
                self.results[model_name] = results
                
                # Update best model
                if results['test_accuracy'] > self.best_score:
                    self.best_score = results['test_accuracy']
                    self.best_model = best_model
                    self.best_model_name = model_name
                
                logger.info(f"✅ {model_name} completed - Accuracy: {results['test_accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        logger.info(f"🏆 Best model: {self.best_model_name} (Accuracy: {self.best_score:.4f})")
        
        return self.results
    
    def _evaluate_model(self, model, X_train, X_test, y_train, y_test, 
                       model_name: str) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Probabilities (if available)
        try:
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_train_proba = y_train_pred
            y_test_proba = y_test_pred
        
        # Calculate metrics
        results = {
            'model_name': model_name,
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'train_precision': precision_score(y_train, y_train_pred, average='weighted'),
            'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
            'train_recall': recall_score(y_train, y_train_pred, average='weighted'),
            'test_recall': recall_score(y_test, y_test_pred, average='weighted'),
            'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
            'test_f1': f1_score(y_test, y_test_pred, average='weighted'),
            'train_auc': roc_auc_score(y_train, y_train_proba),
            'test_auc': roc_auc_score(y_test, y_test_proba)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
        
        return results
    
    def save_models(self) -> Dict[str, str]:
        """Save all trained models"""
        logger.info("💾 Saving trained models...")
        
        saved_paths = {}
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = self.models_dir / f"csk_{model_name}_model.pkl"
            joblib.dump(model, model_path)
            saved_paths[model_name] = str(model_path)
            logger.info(f"✅ Saved {model_name} to {model_path}")
        
        # Save best model separately
        if self.best_model:
            best_model_path = self.models_dir / "csk_best_model.pkl"
            joblib.dump(self.best_model, best_model_path)
            saved_paths['best_model'] = str(best_model_path)
            logger.info(f"✅ Saved best model ({self.best_model_name}) to {best_model_path}")
        
        # Save results
        results_path = self.models_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        saved_paths['results'] = str(results_path)
        
        # Save model metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'training_date': datetime.now().isoformat(),
            'models_trained': list(self.models.keys()),
            'total_models': len(self.models)
        }
        
        metadata_path = self.models_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_paths['metadata'] = str(metadata_path)
        
        logger.info("✅ All models and results saved successfully")
        return saved_paths
    
    def get_feature_importance(self, model_name: str = None) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return {}
        
        model = self.models[model_name]
        
        try:
            # Get feature importance from the classifier in pipeline
            if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                classifier = model.named_steps['classifier']
                if hasattr(classifier, 'feature_importances_'):
                    return dict(enumerate(classifier.feature_importances_))
                elif hasattr(classifier, 'coef_'):
                    return dict(enumerate(classifier.coef_[0]))
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def create_results_summary(self) -> pd.DataFrame:
        """Create comprehensive results summary"""
        if not self.results:
            logger.warning("No results available")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results).T
        
        # Sort by test accuracy
        results_df = results_df.sort_values('test_accuracy', ascending=False)
        
        # Add ranking
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # Calculate overfitting indicator
        results_df['overfitting'] = results_df['train_accuracy'] - results_df['test_accuracy']
        
        return results_df

# Convenience functions
def train_csk_models(X: pd.DataFrame, y: pd.Series) -> CSKModelTrainer:
    """Quick function to train all CSK models"""
    trainer = CSKModelTrainer()
    trainer.train_all_models(X, y)
    trainer.save_models()
    return trainer

def load_best_model(models_dir: str = "models"):
    """Load the best trained model"""
    model_path = Path(models_dir) / "csk_best_model.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Best model not found at {model_path}")

def get_training_results(models_dir: str = "models") -> Dict[str, Any]:
    """Load training results"""
    results_path = Path(models_dir) / "training_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Training results not found at {results_path}")
