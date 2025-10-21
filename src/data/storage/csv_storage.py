import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any
from utils.config import Config

class CSVStorage:
    """Handles storage of collected data in CSV format."""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or Config.DATA_STORAGE_PATH
        self._ensure_storage_directory()
    
    def _ensure_storage_directory(self):
        """Create storage directory if it doesn't exist."""
        os.makedirs(self.storage_path, exist_ok=True)


    def save_data(self, data: List[Dict[str, Any]], brand_name: str, platform: str) -> str:
        if not data:
            print("⚠️ No data to save")  # Better logging
            return None
    
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{brand_name}_{platform}_{timestamp}.csv"
        filepath = os.path.join(self.storage_path, filename)
    
        try:
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            print(f"💾 Saved {len(data)} items to {filename}")  # Success feedback
            return filepath
        except Exception as e:
            print(f"❌ Error saving data to {filename}: {str(e)}")  # Better error context
            return None


    
    def read_latest_data(self, brand_name: str, platform: str) -> pd.DataFrame:
        try:
            pattern = f"{brand_name}_{platform}_"
            files = [f for f in os.listdir(self.storage_path) 
                    if f.startswith(pattern) and f.endswith('.csv')]
        
            if not files:
                print(f"📭 No data files found for {brand_name} on {platform}")
                return None
        
            # Get the most recent file
            latest_file = sorted(files)[-1]
            filepath = os.path.join(self.storage_path, latest_file)
        
            print(f"📂 Loading data from {latest_file}")  # ← Helpful log
            return pd.read_csv(filepath)
        
        except Exception as e:
            print(f"❌ Error reading data: {str(e)}")
            return None
    
   