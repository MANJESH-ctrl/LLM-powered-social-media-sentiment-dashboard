import praw
from typing import List, Dict, Any
import time
from .base_collector import BaseDataCollector
from utils.config import Config
from datetime import datetime

class RedditCollector(BaseDataCollector):
    """Data collector for Reddit platform with enhanced keyword strategies."""
    
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
        Collect posts from Reddit using smart keyword strategies.
        
        Args:
            brand_name: Brand name to search for
            keywords: Additional keywords (comma-separated)
            time_period: "all", "year", "month", "week", "day"
            limit: Number of posts to collect
            
        Returns:
            List of post data dictionaries
        """
        return self.collect_data_with_strategies(brand_name, keywords, time_period, limit)
    
    def collect_data_with_strategies(self, brand_name: str, keywords: str, time_period: str = "all", limit: int = 100) -> List[Dict[str, Any]]:
        """Try multiple collection strategies until we get data"""
        
        strategies = [
            # Strategy 1: Brand + keywords + emotional indicators (most specific)
            {
                "name": "Brand with keywords and emotional indicators",
                "query": self._build_search_query(brand_name, keywords, include_emotional=True),
                "use_keywords": True
            },
            # Strategy 2: Brand + emotional indicators only (fallback)
            {
                "name": "Brand with emotional indicators only", 
                "query": self._build_fallback_query(brand_name),
                "use_keywords": False
            },
            # Strategy 3: Brand only (last resort)
            {
                "name": "Brand only",
                "query": f'"{brand_name}"',
                "use_keywords": False
            }
        ]
        
        collected_data = []
        
        for strategy in strategies:
            if len(collected_data) >= limit:
                break
                
            try:
                print(f"🔄 Trying strategy: {strategy['name']}")
                print(f"   Query: {strategy['query']}")
                
                strategy_data = self._execute_search(
                    strategy['query'], 
                    brand_name, 
                    time_period, 
                    limit - len(collected_data)
                )
                
                if strategy_data:
                    collected_data.extend(strategy_data)
                    print(f"✅ Strategy successful: collected {len(strategy_data)} items")
                    
                    # If we got good data with this strategy, we're done
                    if len(strategy_data) >= 5:  # If we got at least 5 items, good enough
                        break
                else:
                    print(f"⚠️ No results with this strategy")
                    
            except Exception as e:
                print(f"❌ Strategy failed: {e}")
                continue
        
        print(f"🎯 Total collected: {len(collected_data)} items using {len(strategies)} strategies")
        return collected_data
    
    def _execute_search(self, query: str, brand_name: str, time_period: str, limit: int) -> List[Dict[str, Any]]:
        """Execute search with given query"""
        collected_data = []
        
        try:
            subreddit = self.reddit.subreddit("all")
            
            for submission in subreddit.search(
                query, 
                sort='relevance', 
                time_filter=time_period, 
                limit=limit
            ):
                post_data = {
                    'platform': 'reddit',
                    'brand_name': brand_name,
                    'title': submission.title,
                    'text': self._clean_text(submission.selftext),
                    'score': submission.score,
                    'upvotes': submission.ups,
                    'downvotes': submission.downs,
                    'num_comments': submission.num_comments,
                    'search_strategy': query  # Track which query found this
                }
                collected_data.append(post_data)
                
                # Respect rate limits
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error in search execution: {str(e)}")
        
        return collected_data
    


    def _build_search_query(self, brand_name: str, keywords: str, include_emotional: bool = True) -> str:
        """Build search query with EXPANDED emotional indicators"""
        # Expanded emotional indicators for better sentiment-rich content
        EMOTIONAL_INDICATORS = [
            "love", "hate", "best", "worst", "awesome", "terrible", "amazing", "awful",
            "disappointing", "brilliant", "rubbish", "fantastic", "horrible", "perfect",
            "broken", "waste of money", "recommend", "avoid", "regret", "sucks", "great", 
            "bad", "excellent", "poor", "beautiful", "ugly", "fast", "slow", "easy", 
            "difficult", "user-friendly", "complicated", "worth it", "overpriced", 
            "cheap", "expensive", "bargain", "rip-off", "happy", "sad", "angry", 
            "frustrated", "delighted", "disgusted", "pleased", "annoyed", "satisfied",
            "unsatisfied", "like", "dislike", "issue", "problem", "bug", "crash"
        ]
    
        # Start with brand name
        query_parts = [brand_name]
    
        # Add user keywords if provided
        if keywords:
            keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
            query_parts.extend(keyword_list)
    
        # Add emotional indicators to find opinionated content
        if include_emotional:
            query_parts.extend(EMOTIONAL_INDICATORS)
    
        # Use OR to cast a wider net for emotional content
        return ' OR '.join(query_parts)
    

    
    
    def _build_fallback_query(self, brand_name: str) -> str:
        """Fallback query using brand and emotional indicators only"""
        emotional_group = " OR ".join(self.EMOTIONAL_INDICATORS)
        return f'"{brand_name}" AND ({emotional_group})'
    
    def _clean_text(self, text: str) -> str:
        """Clean text data for analysis."""
        if not text:
            return ""
        # Remove excessive newlines and whitespace
        return ' '.join(text.split())