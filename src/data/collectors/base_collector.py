from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

class BaseDataCollector(ABC):
    """Abstract base class for all data collectors."""
    
    # Emotional indicators for finding opinionated content
    EMOTIONAL_INDICATORS = [
        "love", "hate", "best", "worst", "awesome", "terrible",
        "amazing", "awful", "disappointing", "brilliant", "rubbish",
        "fantastic", "horrible", "perfect", "broken", "waste of money",
        "recommend", "avoid", "regret", "sucks", "great", "bad"
    ]
    
    def __init__(self, platform_name: str):
        self.platform_name = platform_name
    
    @abstractmethod
    def collect_data(self, brand_name: str, keywords: str, time_period: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Collect data from the platform.
        
        Args:
            brand_name: Name of the brand to analyze
            keywords: Additional keywords for search
            time_period: Time period for data collection
            limit: Maximum number of items to collect
            
        Returns:
            List of dictionaries containing collected data
        """
        pass
    
    def format_timestamp(self, timestamp) -> str:
        """Convert various timestamp formats to ISO format."""
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp).isoformat()
        elif isinstance(timestamp, str):
            return timestamp
        elif hasattr(timestamp, 'isoformat'):
            return timestamp.isoformat()
        else:
            return str(timestamp)
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning - can be overridden by subclasses."""
        if not text:
            return ""
        return ' '.join(text.split())