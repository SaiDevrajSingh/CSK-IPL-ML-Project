"""
Model Prediction Module
Production-ready prediction pipeline for CSK match outcomes
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
import logging
from pathlib import Path
import joblib
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSKPredictor:
    """Production prediction pipeline for CSK matches"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.metadata = None
        self.feature_names = None
        
        # Load model and metadata
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and metadata"""
        try:
            # Load best model
            model_path = self.models_dir / "csk_best_model.pkl"
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"✅ Model loaded from {model_path}")
            else:
                # Try alternative paths
                alternative_paths = [
                    self.models_dir / "artifacts" / "csk_best_model_random_forest.pkl",
                    self.models_dir / "csk_random_forest_model.pkl"
                ]
                
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        self.model = joblib.load(alt_path)
                        logger.info(f"✅ Model loaded from {alt_path}")
                        break
                else:
                    logger.error("❌ No model file found")
                    return False
            
            # Load metadata
            metadata_path = self.models_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info("✅ Metadata loaded")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def prepare_features(self, match_data: Dict[str, Any]) -> np.ndarray:
        """Prepare features from match data for prediction"""
        try:
            # Default feature template
            features = {
                'season': match_data.get('season', 2025),
                'is_home_venue': self._is_home_venue(match_data.get('venue', ''), match_data.get('city', '')),
                'csk_won_toss': int(match_data.get('toss_winner', '') == 'Chennai Super Kings'),
                'chose_to_bat': int(match_data.get('toss_decision', '') == 'bat'),
                'is_peak_season': self._is_peak_season(match_data.get('season', 2025)),
                'season_experience': max(0, match_data.get('season', 2025) - 2008),
                'strong_opponent': self._is_strong_opponent(match_data.get('opponent', '')),
                'is_playoff': self._is_playoff_match(match_data.get('stage', '')),
                'match_number': match_data.get('match_number', 1),
                'venue_encoded': self._encode_venue(match_data.get('venue', '')),
                'opponent_encoded': self._encode_opponent(match_data.get('opponent', ''))
            }
            
            # Create feature vector
            feature_vector = list(features.values())
            
            return np.array(feature_vector).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"❌ Error preparing features: {str(e)}")
            # Return default feature vector
            return np.zeros((1, 11))
    
    def _is_home_venue(self, venue: str, city: str) -> int:
        """Check if match is at home venue"""
        home_indicators = ['chennai', 'chepauk', 'ma chidambaram']
        venue_lower = venue.lower()
        city_lower = city.lower()
        
        return int(any(indicator in venue_lower or indicator in city_lower 
                      for indicator in home_indicators))
    
    def _is_peak_season(self, season: int) -> int:
        """Check if season is a peak performance season for CSK"""
        peak_seasons = [2010, 2011, 2018, 2021, 2023]
        return int(season in peak_seasons)
    
    def _is_strong_opponent(self, opponent: str) -> int:
        """Check if opponent is considered strong"""
        strong_opponents = [
            'Mumbai Indians', 'Royal Challengers Bangalore',
            'Kolkata Knight Riders', 'Delhi Capitals'
        ]
        return int(opponent in strong_opponents)
    
    def _is_playoff_match(self, stage: str) -> int:
        """Check if match is a playoff match"""
        playoff_stages = ['qualifier1', 'qualifier2', 'eliminator', 'final']
        return int(stage.lower() in playoff_stages)
    
    def _encode_venue(self, venue: str) -> int:
        """Simple venue encoding"""
        venue_mapping = {
            'MA Chidambaram Stadium, Chepauk': 0,
            'Wankhede Stadium': 1,
            'Eden Gardens': 2,
            'M Chinnaswamy Stadium': 3,
            'Rajiv Gandhi International Stadium': 4,
            'Sawai Mansingh Stadium': 5,
            'Feroz Shah Kotla': 6,
            'Punjab Cricket Association Stadium': 7
        }
        return venue_mapping.get(venue, 0)
    
    def _encode_opponent(self, opponent: str) -> int:
        """Simple opponent encoding"""
        opponent_mapping = {
            'Mumbai Indians': 0,
            'Royal Challengers Bangalore': 1,
            'Kolkata Knight Riders': 2,
            'Delhi Capitals': 3,
            'Rajasthan Royals': 4,
            'Punjab Kings': 5,
            'Sunrisers Hyderabad': 6,
            'Gujarat Titans': 7,
            'Lucknow Super Giants': 8
        }
        return opponent_mapping.get(opponent, 0)
    
    def predict(self, match_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[int, List[int]]:
        """Make match outcome predictions"""
        try:
            if self.model is None:
                logger.error("❌ Model not loaded")
                return 0 if isinstance(match_data, dict) else [0] * len(match_data)
            
            # Handle single prediction
            if isinstance(match_data, dict):
                features = self.prepare_features(match_data)
                prediction = self.model.predict(features)[0]
                return int(prediction)
            
            # Handle batch predictions
            elif isinstance(match_data, list):
                predictions = []
                for match in match_data:
                    features = self.prepare_features(match)
                    prediction = self.model.predict(features)[0]
                    predictions.append(int(prediction))
                return predictions
            
            else:
                raise ValueError("match_data must be dict or list of dicts")
                
        except Exception as e:
            logger.error(f"❌ Error in prediction: {str(e)}")
            return 0 if isinstance(match_data, dict) else [0] * len(match_data)
    
    def predict_proba(self, match_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Get prediction probabilities"""
        try:
            if self.model is None:
                logger.error("❌ Model not loaded")
                default_proba = np.array([0.5, 0.5])
                return default_proba if isinstance(match_data, dict) else [default_proba] * len(match_data)
            
            # Handle single prediction
            if isinstance(match_data, dict):
                features = self.prepare_features(match_data)
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(features)[0]
                else:
                    # Fallback for models without predict_proba
                    prediction = self.model.predict(features)[0]
                    probabilities = np.array([1 - prediction, prediction])
                return probabilities
            
            # Handle batch predictions
            elif isinstance(match_data, list):
                probabilities = []
                for match in match_data:
                    features = self.prepare_features(match)
                    if hasattr(self.model, 'predict_proba'):
                        proba = self.model.predict_proba(features)[0]
                    else:
                        prediction = self.model.predict(features)[0]
                        proba = np.array([1 - prediction, prediction])
                    probabilities.append(proba)
                return probabilities
            
            else:
                raise ValueError("match_data must be dict or list of dicts")
                
        except Exception as e:
            logger.error(f"❌ Error in probability prediction: {str(e)}")
            default_proba = np.array([0.5, 0.5])
            return default_proba if isinstance(match_data, dict) else [default_proba] * len(match_data)
    
    def get_prediction_explanation(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed prediction explanation"""
        try:
            # Make prediction
            prediction = self.predict(match_data)
            probabilities = self.predict_proba(match_data)
            
            # Prepare features for analysis
            features = self.prepare_features(match_data)
            
            # Create explanation
            explanation = {
                'prediction': 'WIN' if prediction == 1 else 'LOSS',
                'confidence': float(max(probabilities)),
                'win_probability': float(probabilities[1]),
                'loss_probability': float(probabilities[0]),
                'key_factors': self._analyze_key_factors(match_data),
                'model_info': {
                    'model_name': self.metadata.get('best_model_name', 'Unknown') if self.metadata else 'Unknown',
                    'training_accuracy': self.metadata.get('best_score', 0) if self.metadata else 0
                }
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"❌ Error creating explanation: {str(e)}")
            return {
                'prediction': 'UNKNOWN',
                'confidence': 0.5,
                'win_probability': 0.5,
                'loss_probability': 0.5,
                'key_factors': {},
                'model_info': {}
            }
    
    def _analyze_key_factors(self, match_data: Dict[str, Any]) -> Dict[str, str]:
        """Analyze key factors affecting the prediction"""
        factors = {}
        
        # Home advantage
        if self._is_home_venue(match_data.get('venue', ''), match_data.get('city', '')):
            factors['home_advantage'] = 'Playing at home venue (Chennai/Chepauk)'
        
        # Toss advantage
        if match_data.get('toss_winner') == 'Chennai Super Kings':
            factors['toss_advantage'] = 'Won the toss'
            if match_data.get('toss_decision') == 'bat':
                factors['toss_decision'] = 'Chose to bat first'
        
        # Peak season
        if self._is_peak_season(match_data.get('season', 2025)):
            factors['peak_season'] = 'Peak performance season for CSK'
        
        # Strong opponent
        if self._is_strong_opponent(match_data.get('opponent', '')):
            factors['strong_opponent'] = f"Facing strong opponent: {match_data.get('opponent', '')}"
        
        # Playoff match
        if self._is_playoff_match(match_data.get('stage', '')):
            factors['playoff_match'] = 'High-stakes playoff match'
        
        return factors
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_loaded': self.model is not None,
            'model_type': str(type(self.model).__name__) if self.model else 'None',
            'metadata': self.metadata or {},
            'models_dir': str(self.models_dir)
        }

# Convenience functions
def predict_csk_match(match_data: Dict[str, Any], models_dir: str = "models") -> Dict[str, Any]:
    """Quick function to predict CSK match outcome"""
    predictor = CSKPredictor(models_dir)
    return predictor.get_prediction_explanation(match_data)

def batch_predict_matches(matches: List[Dict[str, Any]], models_dir: str = "models") -> List[Dict[str, Any]]:
    """Predict multiple matches"""
    predictor = CSKPredictor(models_dir)
    results = []
    
    for match in matches:
        result = predictor.get_prediction_explanation(match)
        results.append(result)
    
    return results
