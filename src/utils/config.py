import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Reddit API Configuration
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'SentimentAnalysisBot/1.0')
    
    # YouTube API Configuration
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
    
    # Data Storage
    DATA_STORAGE_PATH = os.getenv('DATA_STORAGE_PATH', './outputs/raw_data')
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configurations are present."""
        errors = []
        
        # Only validate Reddit if we're using it
        if not cls.REDDIT_CLIENT_ID:
            errors.append("REDDIT_CLIENT_ID is missing")
        if not cls.REDDIT_CLIENT_SECRET:
            errors.append("REDDIT_CLIENT_SECRET is missing")
        if not cls.YOUTUBE_API_KEY:
            errors.append("YOUTUBE_API_KEY is missing")
            
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")




