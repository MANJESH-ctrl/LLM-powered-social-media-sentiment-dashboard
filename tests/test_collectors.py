import sys
import os

def test_collectors():
    """Test if data collectors can be initialized"""
    print("🔍 TESTING DATA COLLECTORS...")
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    try:
        from src.utils.config import Config
        
        # Test Reddit Collector
        try:
            from src.data.collectors.reddit_collector import RedditCollector
            collector = RedditCollector()
            print("✅ RedditCollector initialized")
        except Exception as e:
            print(f"❌ RedditCollector failed: {e}")
        
        # Test YouTube Collector
        try:
            from src.data.collectors.youtube_collector import YouTubeCollector
            collector = YouTubeCollector()
            print("✅ YouTubeCollector initialized")
        except Exception as e:
            print(f"❌ YouTubeCollector failed: {e}")
            
        # Test CSV Storage
        try:
            from src.data.storage.csv_storage import CSVStorage
            storage = CSVStorage()
            print("✅ CSVStorage initialized")
        except Exception as e:
            print(f"❌ CSVStorage failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Collector test setup failed: {e}")
        return False

if __name__ == "__main__":
    test_collectors()