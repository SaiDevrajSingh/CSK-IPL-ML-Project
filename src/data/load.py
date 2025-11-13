"""
Data Loading Module
Handles saving processed data to various destinations
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Load processed data to storage destinations"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def save_interim_data(self, df: pd.DataFrame, filename: str, 
                         metadata: Optional[Dict] = None) -> str:
        """Save intermediate processed data"""
        try:
            filepath = self.interim_dir / filename
            
            # Save data
            if filename.endswith('.csv'):
                df.to_csv(filepath, index=False)
            elif filename.endswith('.parquet'):
                df.to_parquet(filepath, index=False)
            else:
                # Default to CSV
                filepath = filepath.with_suffix('.csv')
                df.to_csv(filepath, index=False)
            
            logger.info(f"✅ Interim data saved: {filepath}")
            
            # Save metadata if provided
            if metadata:
                metadata_path = filepath.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                logger.info(f"✅ Metadata saved: {metadata_path}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Error saving interim data: {str(e)}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame, filename: str,
                           metadata: Optional[Dict] = None) -> str:
        """Save final processed data ready for modeling"""
        try:
            filepath = self.processed_dir / filename
            
            # Save data
            if filename.endswith('.csv'):
                df.to_csv(filepath, index=False)
            elif filename.endswith('.parquet'):
                df.to_parquet(filepath, index=False)
            else:
                # Default to CSV
                filepath = filepath.with_suffix('.csv')
                df.to_csv(filepath, index=False)
            
            logger.info(f"✅ Processed data saved: {filepath}")
            
            # Save metadata
            data_info = {
                'filename': filename,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'created_at': pd.Timestamp.now().isoformat(),
                'missing_values': df.isnull().sum().to_dict()
            }
            
            if metadata:
                data_info.update(metadata)
            
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(data_info, f, indent=2, default=str)
            
            logger.info(f"✅ Metadata saved: {metadata_path}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Error saving processed data: {str(e)}")
            raise
    
    def save_feature_engineered_data(self, df: pd.DataFrame, 
                                   feature_info: Dict[str, Any]) -> str:
        """Save feature engineered dataset with feature information"""
        filename = "csk_features_engineered.csv"
        
        metadata = {
            'type': 'feature_engineered',
            'feature_count': len(df.columns),
            'feature_info': feature_info,
            'preprocessing_steps': [
                'data_cleaning',
                'feature_engineering',
                'encoding',
                'scaling'
            ]
        }
        
        return self.save_processed_data(df, filename, metadata)
    
    def save_model_ready_data(self, X: pd.DataFrame, y: pd.Series,
                             feature_names: list, target_name: str) -> Dict[str, str]:
        """Save model-ready features and target"""
        try:
            # Save features
            X_path = self.save_processed_data(
                X, 
                "csk_model_features.csv",
                {
                    'type': 'model_features',
                    'feature_names': feature_names,
                    'target_name': target_name
                }
            )
            
            # Save target
            y_df = pd.DataFrame({target_name: y})
            y_path = self.save_processed_data(
                y_df,
                "csk_model_target.csv",
                {
                    'type': 'model_target',
                    'target_name': target_name,
                    'class_distribution': y.value_counts().to_dict()
                }
            )
            
            logger.info("✅ Model-ready data saved successfully")
            
            return {
                'features_path': X_path,
                'target_path': y_path
            }
            
        except Exception as e:
            logger.error(f"❌ Error saving model-ready data: {str(e)}")
            raise
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed data from storage"""
        try:
            filepath = self.processed_dir / filename
            
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            if filename.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            else:
                df = pd.read_csv(filepath)
            
            logger.info(f"✅ Loaded processed data: {filepath} - Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading processed data: {str(e)}")
            raise
    
    def get_data_catalog(self) -> Dict[str, Any]:
        """Get catalog of all available processed datasets"""
        catalog = {
            'interim': [],
            'processed': []
        }
        
        # Scan interim directory
        for file_path in self.interim_dir.glob('*.csv'):
            metadata_path = file_path.with_suffix('.json')
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            catalog['interim'].append({
                'filename': file_path.name,
                'path': str(file_path),
                'size_mb': file_path.stat().st_size / (1024 * 1024),
                'modified': file_path.stat().st_mtime,
                'metadata': metadata
            })
        
        # Scan processed directory
        for file_path in self.processed_dir.glob('*.csv'):
            metadata_path = file_path.with_suffix('.json')
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            catalog['processed'].append({
                'filename': file_path.name,
                'path': str(file_path),
                'size_mb': file_path.stat().st_size / (1024 * 1024),
                'modified': file_path.stat().st_mtime,
                'metadata': metadata
            })
        
        return catalog

# Convenience functions
def save_csk_data(df: pd.DataFrame, stage: str = "processed") -> str:
    """Quick function to save CSK data"""
    loader = DataLoader()
    
    if stage == "interim":
        return loader.save_interim_data(df, "csk_interim_data.csv")
    else:
        return loader.save_processed_data(df, "csk_processed_data.csv")

def load_csk_data(stage: str = "processed") -> pd.DataFrame:
    """Quick function to load CSK data"""
    loader = DataLoader()
    
    if stage == "interim":
        return loader.load_processed_data("csk_interim_data.csv")
    else:
        return loader.load_processed_data("csk_processed_data.csv")
