"""
ML Pipeline Module
Complete machine learning pipeline from data to deployed model
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from pathlib import Path

# Import our custom modules
from ..features.build_features import CSKFeatureEngineer
from ..models.train_model import CSKModelTrainer
from ..models.predict_model import CSKPredictor
from ..data.load import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSKMLPipeline:
    """Complete ML pipeline for CSK match prediction"""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        
        self.loader = DataLoader(data_dir)
        self.feature_engineer = CSKFeatureEngineer()
        self.trainer = CSKModelTrainer(models_dir)
        self.predictor = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
    
    def run_full_ml_pipeline(self, target_col: str = 'csk_won_match') -> Dict[str, Any]:
        """Run the complete ML pipeline"""
        logger.info("🚀 Starting complete ML pipeline...")
        
        try:
            # Load data
            data = self.load_data()
            
            # Feature engineering
            X, y = self.engineer_features(data, target_col)
            
            # Train models
            results = self.train_models(X, y)
            
            # Evaluate and save
            evaluation = self.evaluate_pipeline()
            
            # Initialize predictor
            self.predictor = CSKPredictor(self.models_dir)
            
            pipeline_results = {
                'training_results': results,
                'evaluation': evaluation,
                'feature_count': X.shape[1],
                'data_size': X.shape[0],
                'best_model': self.trainer.best_model_name,
                'best_score': self.trainer.best_score
            }
            
            logger.info("🎉 ML pipeline completed successfully!")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"❌ ML pipeline failed: {str(e)}")
            raise
    
    def load_data(self) -> pd.DataFrame:
        """Load processed data for ML"""
        logger.info("📥 Loading processed data...")
        
        try:
            data = self.loader.load_processed_data("csk_match_level_data.csv")
            logger.info(f"✅ Data loaded: {data.shape}")
            return data
            
        except FileNotFoundError:
            logger.error("❌ Processed data not found. Run ETL pipeline first.")
            raise
    
    def engineer_features(self, data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Engineer features for ML"""
        logger.info("🔧 Engineering features...")
        
        # Create comprehensive features
        data_with_features = self.feature_engineer.create_all_features(data)
        
        # Separate features and target
        if target_col not in data_with_features.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        y = data_with_features[target_col]
        X = data_with_features.drop(columns=[target_col])
        
        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        X_encoded = self.feature_engineer.encode_categorical_features(X, categorical_cols)
        
        # Select numerical features for modeling
        numerical_cols = X_encoded.select_dtypes(include=[np.number]).columns.tolist()
        X_final = X_encoded[numerical_cols]
        
        # Handle missing values
        X_final = X_final.fillna(X_final.median())
        
        # Store feature names
        self.feature_names = X_final.columns.tolist()
        
        logger.info(f"✅ Feature engineering completed: {X_final.shape}")
        return X_final, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train ML models"""
        logger.info("🤖 Training ML models...")
        
        # Train all models
        results = self.trainer.train_all_models(X, y)
        
        # Save models
        saved_paths = self.trainer.save_models()
        
        # Create results summary
        results_summary = self.trainer.create_results_summary()
        
        logger.info("✅ Model training completed")
        
        return {
            'individual_results': results,
            'results_summary': results_summary.to_dict(),
            'saved_paths': saved_paths
        }
    
    def evaluate_pipeline(self) -> Dict[str, Any]:
        """Evaluate the complete pipeline"""
        logger.info("📊 Evaluating ML pipeline...")
        
        evaluation = {}
        
        try:
            # Model performance evaluation
            if self.trainer.results:
                best_result = max(self.trainer.results.values(), key=lambda x: x['test_accuracy'])
                
                evaluation['best_model_performance'] = {
                    'model_name': best_result['model_name'],
                    'test_accuracy': best_result['test_accuracy'],
                    'test_f1': best_result['test_f1'],
                    'test_auc': best_result['test_auc'],
                    'cv_mean': best_result['cv_mean'],
                    'cv_std': best_result['cv_std']
                }
                
                # Check for overfitting
                overfitting = best_result['train_accuracy'] - best_result['test_accuracy']
                evaluation['overfitting_check'] = {
                    'overfitting_score': overfitting,
                    'is_overfitting': overfitting > 0.1,
                    'severity': 'high' if overfitting > 0.2 else 'medium' if overfitting > 0.1 else 'low'
                }
            
            # Feature importance analysis
            if self.trainer.best_model:
                feature_importance = self.trainer.get_feature_importance()
                if feature_importance:
                    # Get top 10 features
                    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                    evaluation['top_features'] = dict(sorted_features[:10])
            
            # Data quality assessment
            evaluation['data_quality'] = {
                'feature_count': len(self.feature_names) if self.feature_names else 0,
                'training_samples': len(self.trainer.results) if self.trainer.results else 0,
                'models_trained': len(self.trainer.models) if self.trainer.models else 0
            }
            
            logger.info("✅ Pipeline evaluation completed")
            
        except Exception as e:
            logger.error(f"❌ Pipeline evaluation failed: {str(e)}")
            evaluation['evaluation_error'] = str(e)
        
        return evaluation
    
    def predict_new_match(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcome for a new match"""
        if self.predictor is None:
            self.predictor = CSKPredictor(self.models_dir)
        
        return self.predictor.get_prediction_explanation(match_data)
    
    def retrain_pipeline(self, new_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Retrain the pipeline with new data"""
        logger.info("🔄 Retraining ML pipeline...")
        
        try:
            # Load existing or new data
            if new_data is not None:
                data = new_data
            else:
                data = self.load_data()
            
            # Re-engineer features
            X, y = self.engineer_features(data, 'csk_won_match')
            
            # Retrain models
            results = self.train_models(X, y)
            
            # Re-evaluate
            evaluation = self.evaluate_pipeline()
            
            # Reinitialize predictor
            self.predictor = CSKPredictor(self.models_dir)
            
            logger.info("✅ Pipeline retraining completed")
            
            return {
                'retraining_results': results,
                'evaluation': evaluation,
                'improvement': self._compare_with_previous_results(results)
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline retraining failed: {str(e)}")
            raise
    
    def _compare_with_previous_results(self, new_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare new results with previous training"""
        # This would compare with stored previous results
        # For now, return placeholder
        return {
            'accuracy_improvement': 0.0,
            'f1_improvement': 0.0,
            'is_better': False
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'data_loaded': self.loader is not None,
            'features_engineered': self.feature_names is not None,
            'models_trained': len(self.trainer.models) > 0 if self.trainer.models else False,
            'best_model_available': self.trainer.best_model is not None,
            'predictor_ready': self.predictor is not None,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'models_count': len(self.trainer.models) if self.trainer.models else 0,
            'best_model_name': self.trainer.best_model_name,
            'best_score': self.trainer.best_score
        }
    
    def export_pipeline_artifacts(self) -> Dict[str, str]:
        """Export all pipeline artifacts"""
        logger.info("📦 Exporting pipeline artifacts...")
        
        artifacts = {}
        
        try:
            # Model artifacts (already saved by trainer)
            artifacts.update(self.trainer.save_models())
            
            # Feature engineering artifacts
            if self.feature_names:
                feature_info = {
                    'feature_names': self.feature_names,
                    'feature_count': len(self.feature_names),
                    'encoders': {name: str(type(encoder)) for name, encoder in self.feature_engineer.encoders.items()},
                    'scalers': {name: str(type(scaler)) for name, scaler in self.feature_engineer.scalers.items()}
                }
                
                feature_path = Path(self.models_dir) / "feature_engineering_info.json"
                import json
                with open(feature_path, 'w') as f:
                    json.dump(feature_info, f, indent=2, default=str)
                artifacts['feature_info'] = str(feature_path)
            
            # Pipeline summary
            pipeline_summary = {
                'pipeline_version': '1.0.0',
                'created_at': pd.Timestamp.now().isoformat(),
                'status': self.get_pipeline_status(),
                'data_dir': self.data_dir,
                'models_dir': self.models_dir
            }
            
            summary_path = Path(self.models_dir) / "pipeline_summary.json"
            import json
            with open(summary_path, 'w') as f:
                json.dump(pipeline_summary, f, indent=2, default=str)
            artifacts['pipeline_summary'] = str(summary_path)
            
            logger.info("✅ Pipeline artifacts exported")
            
        except Exception as e:
            logger.error(f"❌ Error exporting artifacts: {str(e)}")
        
        return artifacts

# Convenience functions
def run_complete_ml_pipeline(data_dir: str = "data", models_dir: str = "models") -> Dict[str, Any]:
    """Quick function to run complete ML pipeline"""
    pipeline = CSKMLPipeline(data_dir, models_dir)
    return pipeline.run_full_ml_pipeline()

def predict_csk_match_outcome(match_data: Dict[str, Any], models_dir: str = "models") -> Dict[str, Any]:
    """Quick function to predict match outcome"""
    pipeline = CSKMLPipeline(models_dir=models_dir)
    return pipeline.predict_new_match(match_data)

def get_ml_pipeline_status(data_dir: str = "data", models_dir: str = "models") -> Dict[str, Any]:
    """Quick function to get pipeline status"""
    pipeline = CSKMLPipeline(data_dir, models_dir)
    return pipeline.get_pipeline_status()
