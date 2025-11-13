"""
Data Extraction Module
Handles loading data from various sources (CSV, APIs, databases)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataExtractor:
    """Extract data from various sources"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.external_dir = self.data_dir / "external"
    
    def load_ipl_data(self, filename: str = "IPL.csv") -> pd.DataFrame:
        """Load IPL dataset from raw data directory"""
        try:
            filepath = self.raw_dir / filename
            logger.info(f"Loading IPL data from: {filepath}")
            
            df = pd.read_csv(filepath, low_memory=False)
            logger.info(f"✅ Data loaded successfully: {df.shape}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"❌ File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise
    
    def load_external_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from external sources"""
        try:
            if source == "cricapi":
                return self._load_from_cricapi(**kwargs)
            elif source == "espn":
                return self._load_from_espn(**kwargs)
            else:
                raise ValueError(f"Unsupported data source: {source}")
                
        except Exception as e:
            logger.error(f"❌ Error loading external data: {str(e)}")
            raise
    
    def _load_from_cricapi(self, api_key: str, match_ids: list) -> pd.DataFrame:
        """Load match data from CricAPI"""
        # Implementation for CricAPI integration
        logger.info("Loading data from CricAPI...")
        # Placeholder implementation
        return pd.DataFrame()
    
    def _load_from_espn(self, match_urls: list) -> pd.DataFrame:
        """Load match data from ESPN Cricinfo"""
        # Implementation for ESPN scraping
        logger.info("Loading data from ESPN Cricinfo...")
        # Placeholder implementation
        return pd.DataFrame()
    
    def validate_data_schema(self, df: pd.DataFrame) -> bool:
        """Validate that loaded data has expected schema"""
        required_columns = [
            'match_id', 'date', 'batting_team', 'bowling_team',
            'venue', 'toss_winner', 'toss_decision', 'match_won_by'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"⚠️ Missing columns: {missing_columns}")
            return False
        
        logger.info("✅ Data schema validation passed")
        return True
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data information"""
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'date_range': {
                'min': df['date'].min() if 'date' in df.columns else None,
                'max': df['date'].max() if 'date' in df.columns else None
            }
        }

    def extract_csk_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract only CSK matches from the dataset"""
        csk_name = "Chennai Super Kings"
        csk_matches = df[
            (df['batting_team'] == csk_name) | 
            (df['bowling_team'] == csk_name)
        ].copy()
        
        logger.info(f"✅ Extracted {len(csk_matches)} CSK match records")
        return csk_matches

# Convenience functions
def load_ipl_data(data_dir: str = "data") -> pd.DataFrame:
    """Quick function to load IPL data"""
    extractor = DataExtractor(data_dir)
    return extractor.load_ipl_data()

def extract_csk_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only CSK matches from the dataset"""
    csk_name = "Chennai Super Kings"
    csk_matches = df[
        (df['batting_team'] == csk_name) | 
        (df['bowling_team'] == csk_name)
    ].copy()
    
    logger.info(f"✅ Extracted {len(csk_matches)} CSK match records")
    return csk_matches
