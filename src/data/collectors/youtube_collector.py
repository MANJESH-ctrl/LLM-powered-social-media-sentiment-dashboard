from googleapiclient.discovery import build
from typing import List, Dict, Any
from .base_collector import BaseDataCollector
from src.utils.config import Config
from datetime import datetime

class YouTubeCollector(BaseDataCollector):
    """Data collector for YouTube platform."""
    
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
        Collect videos and comments from YouTube.
        
        Args:
            brand_name: Brand name to search for
            keywords: Additional keywords (comma-separated)
            time_period: Not fully implemented for YouTube in initial version
            limit: Number of videos to collect
            
        Returns:
            List of video and comment data dictionaries
        """
        search_query = self._build_search_query(brand_name, keywords)
        collected_data = []
        
        try:
            # Search for videos
            search_response = self.youtube.search().list(
                q=search_query,
                part='snippet',
                type='video',
                maxResults=min(limit, 50),  # YouTube API limit
                order='relevance'
            ).execute()
            
            for item in search_response.get('items', []):
                video_data = self._extract_video_data(item, brand_name)
                if video_data:
                    collected_data.append(video_data)
                
                # Get comments for the video
                comments = self._get_video_comments(item['id']['videoId'], brand_name)
                collected_data.extend(comments)
                
        except Exception as e:
            print(f"Error collecting YouTube data: {str(e)}")
        
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
                'id': video_id,
                'title': snippet['title'],
                'text': self._clean_text(snippet['description']),
                'author': snippet['channelTitle'],
                'created_utc': snippet['publishedAt'],
                'views': int(stats.get('viewCount', 0)),
                'likes': int(stats.get('likeCount', 0)),
                'comments_count': int(stats.get('commentCount', 0)),
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'content_type': 'video',
                'collected_at': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error extracting video data: {str(e)}")
            return None
    
    def _get_video_comments(self, video_id: str, brand_name: str, max_comments: int = 20) -> List[Dict[str, Any]]:
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
                    'id': item['id'],
                    'title': '',  # Comments don't have titles
                    'text': self._clean_text(comment['textDisplay']),
                    'author': comment['authorDisplayName'],
                    'created_utc': comment['publishedAt'],
                    'likes': int(comment.get('likeCount', 0)),
                    'views': 0,
                    'comments_count': 0,
                    'url': f"https://www.youtube.com/watch?v={video_id}&lc={item['id']}",
                    'content_type': 'comment',
                    'collected_at': datetime.now().isoformat()
                }
                comments_data.append(comment_data)
                
        except Exception as e:
            # Comments might be disabled
            print(f"Error getting comments for video {video_id}: {str(e)}")
        
        return comments_data
    
    def _build_search_query(self, brand_name: str, keywords: str) -> str:
        """Build search query from brand name and keywords."""
        query_parts = [brand_name]
        if keywords:
            query_parts.extend([kw.strip() for kw in keywords.split(',')])
        return ' '.join(query_parts)
    
    def _clean_text(self, text: str) -> str:
        """Clean text data for analysis."""
        if not text:
            return ""
        # Remove HTML tags and excessive whitespace
        import re
        text = re.sub(r'<[^>]+>', '', text)
        return ' '.join(text.split())