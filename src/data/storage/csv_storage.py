import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any
from src.utils.config import Config

class CSVStorage:
    """Handles storage of collected data in CSV format."""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or Config.DATA_STORAGE_PATH
        self._ensure_storage_directory()
    
    def _ensure_storage_directory(self):
        """Create storage directory if it doesn't exist."""
        os.makedirs(self.storage_path, exist_ok=True)
    
    def save_data(self, data: List[Dict[str, Any]], brand_name: str, platform: str) -> str:
        """
        Save collected data to CSV file.
        
        Args:
            data: List of data dictionaries
            brand_name: Name of the brand
            platform: Platform name ('reddit' or 'youtube')
            
        Returns:
            Path to the saved CSV file
        """
        if not data:
            return None
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{brand_name}_{platform}_{timestamp}.csv"
        filepath = os.path.join(self.storage_path, filename)
        
        try:
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            return filepath
        except Exception as e:
            print(f"Error saving data to CSV: {str(e)}")
            return None
    
    def read_latest_data(self, brand_name: str, platform: str) -> pd.DataFrame:
        """Read the most recent data file for a brand and platform."""
        try:
            # Look for files matching the pattern
            pattern = f"{brand_name}_{platform}_"
            files = [f for f in os.listdir(self.storage_path) if f.startswith(pattern) and f.endswith('.csv')]
            
            if not files:
                return None
            
            # Get the most recent file
            latest_file = sorted(files)[-1]
            filepath = os.path.join(self.storage_path, latest_file)
            
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading data: {str(e)}")
            return None