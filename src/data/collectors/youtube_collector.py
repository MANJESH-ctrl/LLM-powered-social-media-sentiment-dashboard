from googleapiclient.discovery import build
from typing import List, Dict, Any
from .base_collector import BaseDataCollector
from utils.config import Config
from datetime import datetime

class YouTubeCollector(BaseDataCollector):
    """Data collector for YouTube platform with enhanced keyword strategies."""
    
    def __init__(self):
        super().__init__("youtube")
        self.setup_client()
    
    def setup_client(self):
        """Initialize YouTube API client."""
        try:
            self.youtube = build('youtube', 'v3', developerKey=Config.YOUTUBE_API_KEY)
        except Exception as e:
            raise Exception(f"Failed to initialize YouTube client: {str(e)}")
    
    def collect_data(self, brand_name: str, keywords: str, time_period: str = "any", limit: int = 50) -> List[Dict[str, Any]]:
        """
        Collect videos and comments from YouTube using smart strategies.
        """
        return self.collect_data_with_strategies(brand_name, keywords, time_period, limit)
    
    def collect_data_with_strategies(self, brand_name: str, keywords: str, time_period: str = "any", limit: int = 50) -> List[Dict[str, Any]]:
        """Try multiple collection strategies until we get data"""
        
        strategies = [
            # Strategy 1: Brand + keywords + emotional indicators
            {
                "name": "Brand with keywords and emotional indicators",
                "query": self._build_search_query(brand_name, keywords, include_emotional=True),
                "use_keywords": True
            },
            # Strategy 2: Brand + emotional indicators only
            {
                "name": "Brand with emotional indicators only", 
                "query": self._build_fallback_query(brand_name),
                "use_keywords": False
            },
            # Strategy 3: Brand only
            {
                "name": "Brand only",
                "query": brand_name,
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
                    limit - len(collected_data)
                )
                
                if strategy_data:
                    collected_data.extend(strategy_data)
                    print(f"✅ Strategy successful: collected {len(strategy_data)} items")
                    
                    # If we got good data with this strategy, we're done
                    if len(strategy_data) >= 3:  # If we got at least 3 items, good enough
                        break
                else:
                    print(f"⚠️ No results with this strategy")
                    
            except Exception as e:
                print(f"❌ Strategy failed: {e}")
                continue
        
        print(f"🎯 Total collected: {len(collected_data)} items using {len(strategies)} strategies")
        return collected_data
    
    def _execute_search(self, query: str, brand_name: str, limit: int) -> List[Dict[str, Any]]:
        """Execute search with given query"""
        collected_data = []
        
        try:
            # Search for videos
            search_response = self.youtube.search().list(
                q=query,
                part='snippet',
                type='video',
                maxResults=min(limit, 50),  # YouTube API limit
                order='relevance'
            ).execute()
            
            for item in search_response.get('items', []):
                video_data = self._extract_video_data(item, brand_name)
                if video_data:
                    video_data['search_strategy'] = query  # Track which query found this
                    collected_data.append(video_data)
                
                # Get comments for the video
                comments = self._get_video_comments(item['id']['videoId'], brand_name, max_comments=5)
                for comment in comments:
                    comment['search_strategy'] = query
                collected_data.extend(comments)
                
        except Exception as e:
            print(f"Error in YouTube search execution: {str(e)}")
        
        return collected_data
    
    def _extract_video_data(self, item: Dict, brand_name: str) -> Dict[str, Any]:
        """Extract relevant data from video search result."""
        try:
            video_id = item['id']['videoId']
            
            # Get video statistics
            video_response = self.youtube.videos().list(
                part='statistics,snippet',
                id=video_id
            ).execute()
            
            if not video_response.get('items'):
                return None
            
            video_info = video_response['items'][0]
            stats = video_info['statistics']
            snippet = video_info['snippet']
            
            return {
                'platform': 'youtube',
                'brand_name': brand_name,
                'title': snippet['title'],
                'text': self._clean_text(snippet['description']),
                'views': int(stats.get('viewCount', 0)),
                'likes': int(stats.get('likeCount', 0)),
                'comments_count': int(stats.get('commentCount', 0)),
                'content_type': 'video',
            }
        except Exception as e:
            print(f"Error extracting video data: {str(e)}")
            return None
    
    def _get_video_comments(self, video_id: str, brand_name: str, max_comments: int = 5) -> List[Dict[str, Any]]:
        """Get comments for a video."""
        comments_data = []
        
        try:
            comments_response = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(max_comments, 20),  # YouTube API limit
                order='relevance'
            ).execute()
            
            for item in comments_response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                
                comment_data = {
                    'platform': 'youtube',
                    'brand_name': brand_name,
                    'title': '',  # Comments don't have titles
                    'text': self._clean_text(comment['textDisplay']),
                    'likes': int(comment.get('likeCount', 0)),
                    'content_type': 'comment',
                }
                comments_data.append(comment_data)
                
        except Exception as e:
            # Comments might be disabled
            print(f"Error getting comments for video {video_id}: {str(e)}")
        
        return comments_data


    def _build_search_query(self, brand_name: str, keywords: str, include_emotional: bool = True) -> str:
        """Build focused search query with AND/OR logic - JUST LIKE REDDIT"""
       
        # Start with exact brand match
        query_parts = [f'"{brand_name}"']
    
        # Add user keywords if provided
        if keywords:
            keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
            if keyword_list:
                # Use OR for keywords within the same group
                keywords_group = " OR ".join(keyword_list)
                query_parts.append(f"({keywords_group})")
    
        # Add emotional indicators to find opinionated content
        if include_emotional:
            emotional_group = " OR ".join(self.EMOTIONAL_INDICATORS)
            query_parts.append(f"({emotional_group})")
    
        # Combine with AND logic for maximum relevance
        return " AND ".join(query_parts)  # YouTube handles AND with spaces

    def _build_fallback_query(self, brand_name: str) -> str:
        """Fallback query using brand and emotional indicators only"""

        emotional_group = " OR ".join(self.EMOTIONAL_INDICATORS)
        return f'"{brand_name}" AND ({emotional_group})'
    
    
    def _clean_text(self, text: str) -> str:
        """Clean text data for analysis."""
        if not text:
            return ""
        # Remove HTML tags and excessive whitespace
        import re
        text = re.sub(r'<[^>]+>', '', text)
        return ' '.join(text.split())