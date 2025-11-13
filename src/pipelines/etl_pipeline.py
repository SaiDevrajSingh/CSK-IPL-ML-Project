"""
ETL Pipeline Module
Complete Extract-Transform-Load pipeline for CSK data processing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path

# Import our custom modules
from ..data.extract import DataExtractor
from ..data.transform import DataTransformer
from ..data.load import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSKETLPipeline:
    """Complete ETL pipeline for CSK match data"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.extractor = DataExtractor(data_dir)
        self.transformer = DataTransformer()
        self.loader = DataLoader(data_dir)
        
        self.raw_data = None
        self.cleaned_data = None
        self.processed_data = None
        self.match_data = None
    
    def run_full_pipeline(self, source_file: str = "IPL.csv") -> Dict[str, str]:
        """Run the complete ETL pipeline"""
        logger.info("🚀 Starting complete ETL pipeline...")
        
        try:
            # Extract
            self.raw_data = self.extract_data(source_file)
            
            # Transform
            self.cleaned_data = self.transform_data(self.raw_data)
            
            # Load
            saved_paths = self.load_data(self.cleaned_data)
            
            logger.info("🎉 ETL pipeline completed successfully!")
            return saved_paths
            
        except Exception as e:
            logger.error(f"❌ ETL pipeline failed: {str(e)}")
            raise
    
    def extract_data(self, source_file: str) -> pd.DataFrame:
        """Extract data from source"""
        logger.info("📥 EXTRACT: Loading raw data...")
        
        # Load IPL data
        df = self.extractor.load_ipl_data(source_file)
        
        # Validate schema
        if not self.extractor.validate_data_schema(df):
            logger.warning("⚠️ Data schema validation failed, continuing anyway...")
        
        # Extract CSK matches only
        csk_data = self.extractor.extract_csk_matches(df)
        
        # Get data info
        data_info = self.extractor.get_data_info(csk_data)
        logger.info(f"✅ EXTRACT completed: {data_info['shape']}")
        
        return csk_data
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform and clean data"""
        logger.info("🔄 TRANSFORM: Processing data...")
        
        # Clean data
        df_clean = self.transformer.clean_data(df)
        
        # Create CSK features
        df_csk = self.transformer.create_csk_features(df_clean)
        
        # Add temporal features
        df_temporal = self.transformer.add_temporal_features(df_csk)
        
        # Create match-level data
        match_data = self.transformer.create_match_level_data(df_temporal)
        self.match_data = match_data
        
        logger.info(f"✅ TRANSFORM completed: {df_temporal.shape}")
        return df_temporal
    
    def load_data(self, df: pd.DataFrame) -> Dict[str, str]:
        """Load processed data to storage"""
        logger.info("💾 LOAD: Saving processed data...")
        
        saved_paths = {}
        
        # Save interim data (ball-by-ball with features)
        interim_path = self.loader.save_interim_data(
            df, 
            "csk_ball_by_ball_processed.csv",
            {
                'type': 'ball_by_ball',
                'processing_steps': ['cleaning', 'feature_creation', 'temporal_features'],
                'csk_matches_only': True
            }
        )
        saved_paths['interim_data'] = interim_path
        
        # Save processed match-level data
        if self.match_data is not None:
            processed_path = self.loader.save_processed_data(
                self.match_data,
                "csk_match_level_data.csv",
                {
                    'type': 'match_level',
                    'ready_for_modeling': True,
                    'target_column': 'csk_won_match'
                }
            )
            saved_paths['processed_data'] = processed_path
        
        logger.info("✅ LOAD completed")
        return saved_paths
    
    def run_incremental_pipeline(self, new_data: pd.DataFrame) -> Dict[str, str]:
        """Run pipeline for new/incremental data"""
        logger.info("🔄 Running incremental ETL pipeline...")
        
        try:
            # Transform new data
            processed_new_data = self.transform_data(new_data)
            
            # Load existing processed data
            try:
                existing_data = self.loader.load_processed_data("csk_match_level_data.csv")
                
                # Combine with new data
                combined_data = pd.concat([existing_data, processed_new_data], ignore_index=True)
                combined_data = combined_data.drop_duplicates(subset=['match_id'], keep='last')
                
            except FileNotFoundError:
                logger.info("No existing data found, using new data only")
                combined_data = processed_new_data
            
            # Save updated data
            saved_paths = self.load_data(combined_data)
            
            logger.info("✅ Incremental pipeline completed")
            return saved_paths
            
        except Exception as e:
            logger.error(f"❌ Incremental pipeline failed: {str(e)}")
            raise
    
    def validate_pipeline_output(self) -> Dict[str, bool]:
        """Validate pipeline output quality"""
        logger.info("🔍 Validating pipeline output...")
        
        validation_results = {}
        
        try:
            # Load processed data
            processed_data = self.loader.load_processed_data("csk_match_level_data.csv")
            
            # Check data quality
            validation_results['data_loaded'] = True
            validation_results['has_target'] = 'csk_won_match' in processed_data.columns
            validation_results['no_missing_target'] = not processed_data['csk_won_match'].isnull().any()
            validation_results['balanced_classes'] = (
                processed_data['csk_won_match'].value_counts().min() / 
                processed_data['csk_won_match'].value_counts().max() > 0.3
            )
            validation_results['sufficient_data'] = len(processed_data) >= 50
            
            # Check feature quality
            feature_cols = [col for col in processed_data.columns if col != 'csk_won_match']
            validation_results['has_features'] = len(feature_cols) > 0
            validation_results['no_all_missing_features'] = not any(
                processed_data[col].isnull().all() for col in feature_cols
            )
            
            logger.info("✅ Pipeline validation completed")
            
        except Exception as e:
            logger.error(f"❌ Pipeline validation failed: {str(e)}")
            validation_results['validation_failed'] = True
        
        return validation_results
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary"""
        summary = {
            'pipeline_status': 'completed' if self.processed_data is not None else 'not_run',
            'data_shapes': {},
            'processing_steps': [
                'data_extraction',
                'data_cleaning',
                'feature_creation',
                'temporal_features',
                'match_level_aggregation'
            ]
        }
        
        if self.raw_data is not None:
            summary['data_shapes']['raw_data'] = self.raw_data.shape
        
        if self.cleaned_data is not None:
            summary['data_shapes']['cleaned_data'] = self.cleaned_data.shape
        
        if self.match_data is not None:
            summary['data_shapes']['match_data'] = self.match_data.shape
            summary['target_distribution'] = self.match_data['csk_won_match'].value_counts().to_dict()
        
        return summary

# Convenience functions
def run_csk_etl_pipeline(source_file: str = "IPL.csv", data_dir: str = "data") -> Dict[str, str]:
    """Quick function to run complete ETL pipeline"""
    pipeline = CSKETLPipeline(data_dir)
    return pipeline.run_full_pipeline(source_file)

def load_processed_csk_data(data_dir: str = "data") -> pd.DataFrame:
    """Quick function to load processed CSK data"""
    loader = DataLoader(data_dir)
    return loader.load_processed_data("csk_match_level_data.csv")

def validate_etl_output(data_dir: str = "data") -> Dict[str, bool]:
    """Quick function to validate ETL output"""
    pipeline = CSKETLPipeline(data_dir)
    return pipeline.validate_pipeline_output()
