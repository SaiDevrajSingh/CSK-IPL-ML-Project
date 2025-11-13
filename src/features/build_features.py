"""
Feature Engineering Module
Advanced feature creation for CSK match prediction
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif

# Setup logging with dd/mm/yyyy format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)

class CSKFeatureEngineer:
    """Advanced feature engineering for CSK match prediction"""
    
    def __init__(self):
        self.csk_name = "Chennai Super Kings"
        self.encoders = {}
        self.scalers = {}
        self.feature_names = []
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        logger.info("🔧 Starting comprehensive feature engineering...")
        
        df_features = df.copy()
        
        # Basic match features
        df_features = self.create_match_features(df_features)
        
        # Performance features
        df_features = self.create_performance_features(df_features)
        
        # Historical features
        df_features = self.create_historical_features(df_features)
        
        # Contextual features
        df_features = self.create_contextual_features(df_features)
        
        # Interaction features
        df_features = self.create_interaction_features(df_features)
        
        logger.info(f"✅ Feature engineering completed! Shape: {df_features.shape}")
        return df_features
    
    def create_match_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic match-level features"""
        logger.info("🏏 Creating match features...")
        
        df_match = df.copy()
        
        # Toss features
        df_match['csk_won_toss'] = (df_match['toss_winner'] == self.csk_name).astype(int)
        df_match['chose_to_bat'] = (df_match['toss_decision'] == 'bat').astype(int)
        df_match['toss_advantage'] = df_match['csk_won_toss'] * df_match['chose_to_bat']
        
        # Venue features
        df_match['is_home_venue'] = df_match['venue'].str.contains(
            'Chennai|Chepauk|MA Chidambaram', case=False, na=False
        ).astype(int)
        
        # Season features
        if 'season' in df_match.columns:
            df_match['season_numeric'] = df_match['season'].astype(str).str.extract('(\d+)').astype(float)
            df_match['season_experience'] = np.maximum(0, df_match['season_numeric'] - 2008)
            
            # Peak seasons
            peak_seasons = [2010, 2011, 2018, 2021, 2023]
            df_match['is_peak_season'] = df_match['season_numeric'].isin(peak_seasons).astype(int)
        
        # Match timing features
        if 'date' in df_match.columns:
            df_match['month'] = pd.to_datetime(df_match['date']).dt.month
            df_match['is_ipl_season'] = df_match['month'].isin([3, 4, 5, 9, 10, 11]).astype(int)
            df_match['day_of_week'] = pd.to_datetime(df_match['date']).dt.dayofweek
            df_match['is_weekend'] = (df_match['day_of_week'] >= 5).astype(int)
        
        logger.info("✅ Match features created")
        return df_match
    
    def create_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create performance-based features"""
        logger.info("📊 Creating performance features...")
        
        df_perf = df.copy()
        
        # Ball-by-ball performance features (if available)
        if 'over' in df_perf.columns and 'ball' in df_perf.columns:
            df_perf['over_ball'] = df_perf['over'] + (df_perf['ball'] / 6)
            df_perf['is_powerplay'] = (df_perf['over'] <= 6).astype(int)
            df_perf['is_middle_overs'] = ((df_perf['over'] > 6) & (df_perf['over'] <= 15)).astype(int)
            df_perf['is_death_overs'] = (df_perf['over'] > 15).astype(int)
        
        # Batting features
        if 'runs_batter' in df_perf.columns and 'balls_faced' in df_perf.columns:
            df_perf['strike_rate'] = (df_perf['runs_batter'] / (df_perf['balls_faced'] + 1e-6)) * 100
            df_perf['is_boundary'] = (df_perf['runs_batter'] >= 4).astype(int)
            df_perf['is_six'] = (df_perf['runs_batter'] == 6).astype(int)
        
        # Bowling features
        if 'runs_bowler' in df_perf.columns and 'bowler_wicket' in df_perf.columns:
            df_perf['economy_rate'] = df_perf['runs_bowler'] / (df_perf['over'] + 1e-6)
            df_perf['is_wicket'] = (df_perf['bowler_wicket'] > 0).astype(int)
        
        # Team performance features
        if 'team_runs' in df_perf.columns and 'team_balls' in df_perf.columns:
            df_perf['team_run_rate'] = (df_perf['team_runs'] / (df_perf['team_balls'] + 1e-6)) * 6
            df_perf['runs_per_wicket'] = df_perf['team_runs'] / (df_perf['team_wicket'] + 1)
        
        logger.info("✅ Performance features created")
        return df_perf
    
    def create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create historical performance features"""
        logger.info("📈 Creating historical features...")
        
        df_hist = df.copy()
        
        # Sort by date for rolling calculations
        if 'date' in df_hist.columns:
            df_hist = df_hist.sort_values('date')
        
        # Rolling win rate (last 5, 10 matches)
        if 'csk_won_match' in df_hist.columns:
            df_hist['rolling_win_rate_5'] = df_hist['csk_won_match'].rolling(
                window=5, min_periods=1
            ).mean()
            df_hist['rolling_win_rate_10'] = df_hist['csk_won_match'].rolling(
                window=10, min_periods=1
            ).mean()
        
        # Head-to-head records (if opponent info available)
        if 'opponent' in df_hist.columns:
            df_hist['h2h_matches'] = df_hist.groupby('opponent').cumcount() + 1
            df_hist['h2h_wins'] = df_hist.groupby('opponent')['csk_won_match'].cumsum()
            df_hist['h2h_win_rate'] = df_hist['h2h_wins'] / df_hist['h2h_matches']
        
        # Venue-specific performance
        if 'venue' in df_hist.columns:
            df_hist['venue_matches'] = df_hist.groupby('venue').cumcount() + 1
            df_hist['venue_wins'] = df_hist.groupby('venue')['csk_won_match'].cumsum()
            df_hist['venue_win_rate'] = df_hist['venue_wins'] / df_hist['venue_matches']
        
        # Season performance
        if 'season' in df_hist.columns:
            df_hist['season_matches'] = df_hist.groupby('season').cumcount() + 1
            df_hist['season_wins'] = df_hist.groupby('season')['csk_won_match'].cumsum()
            df_hist['season_win_rate'] = df_hist['season_wins'] / df_hist['season_matches']
        
        logger.info("✅ Historical features created")
        return df_hist
    
    def create_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create contextual match features"""
        logger.info("🎯 Creating contextual features...")
        
        df_context = df.copy()
        
        # Opponent strength classification
        if 'opponent' in df_context.columns:
            strong_opponents = [
                'Mumbai Indians', 'Royal Challengers Bangalore',
                'Kolkata Knight Riders', 'Delhi Capitals'
            ]
            df_context['strong_opponent'] = df_context['opponent'].isin(strong_opponents).astype(int)
        
        # Match importance
        if 'stage' in df_context.columns:
            playoff_stages = ['qualifier1', 'qualifier2', 'eliminator', 'final']
            df_context['is_playoff'] = df_context['stage'].isin(playoff_stages).astype(int)
        
        # Match number in season
        if 'match_number' in df_context.columns:
            df_context['match_number_norm'] = df_context['match_number'] / 14  # Normalize by typical season length
            df_context['is_early_season'] = (df_context['match_number'] <= 4).astype(int)
            df_context['is_late_season'] = (df_context['match_number'] >= 11).astype(int)
        
        # Pressure situations
        df_context['high_pressure'] = (
            df_context.get('is_playoff', 0) | 
            df_context.get('is_late_season', 0)
        ).astype(int)
        
        logger.info("✅ Contextual features created")
        return df_context
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        logger.info("🔗 Creating interaction features...")
        
        df_interact = df.copy()
        
        # Toss and venue interaction
        if 'csk_won_toss' in df_interact.columns and 'is_home_venue' in df_interact.columns:
            df_interact['toss_home_advantage'] = (
                df_interact['csk_won_toss'] * df_interact['is_home_venue']
            )
        
        # Season experience and opponent strength
        if 'season_experience' in df_interact.columns and 'strong_opponent' in df_interact.columns:
            df_interact['experience_vs_strong'] = (
                df_interact['season_experience'] * df_interact['strong_opponent']
            )
        
        # Home advantage and match importance
        if 'is_home_venue' in df_interact.columns and 'is_playoff' in df_interact.columns:
            df_interact['home_playoff_advantage'] = (
                df_interact['is_home_venue'] * df_interact['is_playoff']
            )
        
        logger.info("✅ Interaction features created")
        return df_interact
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  categorical_cols: List[str]) -> pd.DataFrame:
        """Encode categorical features"""
        logger.info("🔤 Encoding categorical features...")
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                # Handle missing values
                df_encoded[col] = df_encoded[col].fillna('Unknown')
                
                # Label encoding for high cardinality
                if df_encoded[col].nunique() > 10:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                        df_encoded[f'{col}_encoded'] = self.encoders[col].fit_transform(
                            df_encoded[col].astype(str)
                        )
                    else:
                        # Handle unseen categories
                        known_categories = set(self.encoders[col].classes_)
                        df_encoded[col] = df_encoded[col].apply(
                            lambda x: x if x in known_categories else 'Unknown'
                        )
                        df_encoded[f'{col}_encoded'] = self.encoders[col].transform(
                            df_encoded[col].astype(str)
                        )
                
                # One-hot encoding for low cardinality
                else:
                    if col not in self.encoders:
                        self.encoders[col] = OneHotEncoder(drop='first', sparse_output=False)
                        encoded_features = self.encoders[col].fit_transform(
                            df_encoded[[col]].astype(str)
                        )
                        feature_names = [f'{col}_{cat}' for cat in self.encoders[col].categories_[0][1:]]
                        
                        for i, name in enumerate(feature_names):
                            df_encoded[name] = encoded_features[:, i]
        
        logger.info("✅ Categorical encoding completed")
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, 
                               numerical_cols: List[str]) -> pd.DataFrame:
        """Scale numerical features"""
        logger.info("📏 Scaling numerical features...")
        
        df_scaled = df.copy()
        
        for col in numerical_cols:
            if col in df_scaled.columns:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df_scaled[f'{col}_scaled'] = self.scalers[col].fit_transform(
                        df_scaled[[col]]
                    ).flatten()
                else:
                    df_scaled[f'{col}_scaled'] = self.scalers[col].transform(
                        df_scaled[[col]]
                    ).flatten()
        
        logger.info("✅ Numerical scaling completed")
        return df_scaled
    
    def select_best_features(self, X: pd.DataFrame, y: pd.Series, 
                           k: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """Select best features using statistical tests"""
        logger.info(f"🎯 Selecting top {k} features...")
        
        # Remove non-numeric columns for feature selection
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=min(k, X_numeric.shape[1]))
        X_selected = selector.fit_transform(X_numeric, y)
        
        # Get selected feature names
        selected_features = X_numeric.columns[selector.get_support()].tolist()
        
        # Create DataFrame with selected features
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        logger.info(f"✅ Selected {len(selected_features)} best features")
        return X_selected_df, selected_features

# Convenience functions
def engineer_csk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Complete feature engineering pipeline"""
    engineer = CSKFeatureEngineer()
    return engineer.create_all_features(df)

def prepare_model_features(df: pd.DataFrame, target_col: str = 'csk_won_match') -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features for modeling"""
    engineer = CSKFeatureEngineer()
    
    # Create all features
    df_features = engineer.create_all_features(df)
    
    # Separate features and target
    if target_col in df_features.columns:
        y = df_features[target_col]
        X = df_features.drop(columns=[target_col])
    else:
        raise ValueError(f"Target column '{target_col}' not found")
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X_encoded = engineer.encode_categorical_features(X, categorical_cols)
    
    # Scale numerical features
    numerical_cols = X_encoded.select_dtypes(include=[np.number]).columns.tolist()
    X_scaled = engineer.scale_numerical_features(X_encoded, numerical_cols)
    
    # Select only scaled/encoded features for modeling
    feature_cols = [col for col in X_scaled.columns if '_encoded' in col or '_scaled' in col]
    X_final = X_scaled[feature_cols]
    
    return X_final, y
