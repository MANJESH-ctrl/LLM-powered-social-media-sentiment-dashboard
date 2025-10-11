import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration management for API keys and settings."""
    
    # Reddit Configuration
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
    
    # YouTube Configuration
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
    
    # Storage Configuration
    DATA_STORAGE_PATH = os.getenv('DATA_STORAGE_PATH', './outputs/raw_data')
    
    @classmethod
    def validate_config(cls):
        """Validate that all required environment variables are set."""
        required_vars = {
            'REDDIT_CLIENT_ID': cls.REDDIT_CLIENT_ID,
            'REDDIT_CLIENT_SECRET': cls.REDDIT_CLIENT_SECRET,
            'REDDIT_USER_AGENT': cls.REDDIT_USER_AGENT,
            'YOUTUBE_API_KEY': cls.YOUTUBE_API_KEY
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")