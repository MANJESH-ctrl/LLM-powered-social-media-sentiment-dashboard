import praw
from typing import List, Dict, Any
import time
from .base_collector import BaseDataCollector
from src.utils.config import Config
from datetime import datetime

class RedditCollector(BaseDataCollector):
    """Data collector for Reddit platform."""
    
    def __init__(self):
        super().__init__("reddit")
        self.setup_client()
    
    def setup_client(self):
        """Initialize Reddit API client."""
        try:
            self.reddit = praw.Reddit(
                client_id=Config.REDDIT_CLIENT_ID,
                client_secret=Config.REDDIT_CLIENT_SECRET,
                user_agent=Config.REDDIT_USER_AGENT
            )
            # Test connection
            self.reddit.user.me()
        except Exception as e:
            raise Exception(f"Failed to initialize Reddit client: {str(e)}")
    
    def collect_data(self, brand_name: str, keywords: str, time_period: str = "all", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Collect posts from Reddit related to brand and keywords.
        
        Args:
            brand_name: Brand name to search for
            keywords: Additional keywords (comma-separated)
            time_period: "all", "year", "month", "week", "day"
            limit: Number of posts to collect
            
        Returns:
            List of post data dictionaries
        """
        search_query = self._build_search_query(brand_name, keywords)
        collected_data = []
        
        try:
            # Search across multiple subreddits
            subreddits = ["all"]  # You can expand this to specific subreddits
            
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                for submission in subreddit.search(
                    search_query, 
                    sort='relevance', 
                    time_filter=time_period, 
                    limit=limit
                ):
                    post_data = {
                        'platform': 'reddit',
                        'brand_name': brand_name,
                        'id': submission.id,
                        'title': submission.title,
                        'text': self._clean_text(submission.selftext),
                        'author': str(submission.author),
                        'created_utc': self.format_timestamp(submission.created_utc),
                        'score': submission.score,
                        'upvotes': submission.ups,
                        'downvotes': submission.downs,
                        'num_comments': submission.num_comments,
                        'url': submission.url,
                        'permalink': submission.permalink,
                        'collected_at': datetime.now().isoformat()
                    }
                    collected_data.append(post_data)
                    
                    # Respect rate limits
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Error collecting Reddit data: {str(e)}")
        
        return collected_data
    
    def _build_search_query(self, brand_name: str, keywords: str) -> str:
        """Build search query from brand name and keywords."""
        query_parts = [brand_name]
        if keywords:
            query_parts.extend([kw.strip() for kw in keywords.split(',')])
        return ' OR '.join(query_parts)
    
    def _clean_text(self, text: str) -> str:
        """Clean text data for analysis."""
        if not text:
            return ""
        # Remove excessive newlines and whitespace
        return ' '.join(text.split())