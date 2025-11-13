"""
Data Transformation Module
Handles data cleaning, preprocessing, and transformation
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

# Setup logging with dd/mm/yyyy format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DataTransformer:
    """Transform and clean raw data"""
    
    def __init__(self):
        self.csk_name = "Chennai Super Kings"
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive data cleaning"""
        logger.info("🧹 Starting data cleaning process...")
        
        df_clean = df.copy()
        
        # Remove unnamed index columns
        unnamed_cols = [col for col in df_clean.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            df_clean = df_clean.drop(columns=unnamed_cols)
            logger.info(f"✅ Removed unnamed columns: {unnamed_cols}")
        
        # Convert date column
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
            logger.info("✅ Converted date column to datetime")
        
        # Normalize string columns
        string_columns = [
            'batting_team', 'bowling_team', 'venue', 'city',
            'toss_winner', 'toss_decision', 'match_won_by'
        ]
        
        for col in string_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()
        
        logger.info("✅ Normalized string columns")
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        
        if removed_duplicates > 0:
            logger.info(f"✅ Removed {removed_duplicates} duplicate rows")
        
        logger.info(f"🎉 Data cleaning completed! Final shape: {df_clean.shape}")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on column type and context"""
        df_filled = df.copy()
        
        # Categorical columns - fill with 'Unknown'
        categorical_cols = ['venue', 'city', 'toss_decision', 'result_type']
        for col in categorical_cols:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna('Unknown')
        
        # Numerical columns - fill with median
        numerical_cols = df_filled.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_filled[col].isnull().sum() > 0:
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val)
        
        logger.info("✅ Handled missing values")
        return df_filled
    
    def create_csk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create CSK-specific features"""
        logger.info("🏏 Creating CSK-specific features...")
        
        df_csk = df.copy()
        
        # CSK involvement indicator
        df_csk['csk_involved'] = (
            (df_csk['batting_team'] == self.csk_name) | 
            (df_csk['bowling_team'] == self.csk_name)
        ).astype(int)
        
        # Opponent identification
        df_csk['opponent'] = df_csk.apply(self._get_opponent, axis=1)
        
        # Toss features
        df_csk['csk_won_toss'] = (df_csk['toss_winner'] == self.csk_name).astype(int)
        
        # Match outcome
        df_csk['csk_won_match'] = (df_csk['match_won_by'] == self.csk_name).astype(int)
        
        # Home advantage
        df_csk['is_home_match'] = df_csk['venue'].str.contains(
            'Chennai|Chepauk|MA Chidambaram', case=False, na=False
        ).astype(int)
        
        logger.info("✅ CSK features created successfully")
        return df_csk
    
    def _get_opponent(self, row) -> str:
        """Get opponent team name for CSK"""
        if row['batting_team'] == self.csk_name:
            return row['bowling_team']
        elif row['bowling_team'] == self.csk_name:
            return row['batting_team']
        else:
            return 'Unknown'
    
    def create_match_level_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert ball-by-ball data to match-level data"""
        logger.info("📊 Creating match-level dataset...")
        
        # Define match-level columns
        match_cols = [
            'match_id', 'date', 'venue', 'city', 'season',
            'toss_winner', 'toss_decision', 'match_won_by',
            'win_outcome', 'result_type'
        ]
        
        # Get unique matches
        available_cols = [col for col in match_cols if col in df.columns]
        match_df = df[available_cols].drop_duplicates(subset=['match_id'])
        
        # Add CSK-specific features
        match_df['csk_won_match'] = (match_df['match_won_by'] == self.csk_name).astype(int)
        match_df['csk_won_toss'] = (match_df['toss_winner'] == self.csk_name).astype(int)
        match_df['chose_to_bat'] = (match_df['toss_decision'] == 'bat').astype(int)
        
        # Add home advantage
        match_df['is_home_match'] = match_df['venue'].str.contains(
            'Chennai|Chepauk|MA Chidambaram', case=False, na=False
        ).astype(int)
        
        logger.info(f"✅ Match-level data created: {match_df.shape}")
        return match_df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        logger.info("📅 Adding temporal features...")
        
        df_temporal = df.copy()
        
        if 'date' in df_temporal.columns:
            df_temporal['year'] = df_temporal['date'].dt.year
            df_temporal['month'] = df_temporal['date'].dt.month
            df_temporal['day_of_week'] = df_temporal['date'].dt.dayofweek
            df_temporal['is_weekend'] = (df_temporal['day_of_week'] >= 5).astype(int)
        
        if 'season' in df_temporal.columns:
            # Extract numeric season
            df_temporal['season_numeric'] = df_temporal['season'].astype(str).str.extract('(\d+)').astype(float)
            
            # Peak seasons for CSK
            peak_seasons = [2010, 2011, 2018, 2021, 2023]
            df_temporal['is_peak_season'] = df_temporal['season_numeric'].isin(peak_seasons).astype(int)
        
        logger.info("✅ Temporal features added")
        return df_temporal
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers using specified method"""
        logger.info(f"🎯 Removing outliers using {method} method...")
        
        df_clean = df.copy()
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        outliers_removed = 0
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outliers_removed += outlier_mask.sum()
                df_clean = df_clean[~outlier_mask]
        
        logger.info(f"✅ Removed {outliers_removed} outliers")
        return df_clean

# Convenience functions
def clean_ipl_data(df: pd.DataFrame) -> pd.DataFrame:
    """Quick function to clean IPL data"""
    transformer = DataTransformer()
    return transformer.clean_data(df)

def prepare_csk_data(df: pd.DataFrame) -> pd.DataFrame:
    """Complete CSK data preparation pipeline"""
    transformer = DataTransformer()
    
    # Clean data
    df_clean = transformer.clean_data(df)
    
    # Create CSK features
    df_csk = transformer.create_csk_features(df_clean)
    
    # Add temporal features
    df_temporal = transformer.add_temporal_features(df_csk)
    
    # Filter only CSK matches
    csk_matches = df_temporal[df_temporal['csk_involved'] == 1].copy()
    
    return csk_matches
