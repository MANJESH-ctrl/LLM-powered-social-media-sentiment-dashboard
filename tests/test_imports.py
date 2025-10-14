import sys
import os

def test_imports():
    """Test if all custom modules can be imported"""
    print("📦 TESTING IMPORTS...")
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    modules_to_test = [
        ('utils.config', 'Config'),
        ('data.collectors.base_collector', 'BaseDataCollector'),
        ('data.collectors.reddit_collector', 'RedditCollector'),
        ('data.collectors.youtube_collector', 'YouTubeCollector'),
        ('data.storage.csv_storage', 'CSVStorage'),
        ('data.processing.data_processor', 'AdvancedDataProcessor'),
        ('models.sentiment_analyzer', 'AdvancedSentimentAnalyzer'),
    ]
    
    failed_imports = []
    
    for module_path, class_name in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            imported_class = getattr(module, class_name)
            print(f"✅ {module_path}.{class_name}")
        except ImportError as e:
            failed_imports.append(f"{module_path}.{class_name}")
            print(f"❌ {module_path}.{class_name} - {e}")
        except AttributeError as e:
            failed_imports.append(f"{module_path}.{class_name}")
            print(f"❌ {module_path}.{class_name} - Class not found: {e}")
    
    if failed_imports:
        print(f"\n🚨 FAILED IMPORTS: {failed_imports}")
        return False
    else:
        print("🎉 All imports successful!")
        return True

if __name__ == "__main__":
    test_imports()