import sys
import os

def test_config():
    """Test if configuration is set up correctly"""
    print("⚙️ TESTING CONFIGURATION...")
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    # Test 1: Check if .env file exists
    env_file = '.env'
    if not os.path.exists(env_file):
        print("❌ .env file not found in project root")
        return False
    else:
        print("✅ .env file found")
    
    # Test 2: Check if required environment variables are set
    required_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'YOUTUBE_API_KEY']
    
    from dotenv import load_dotenv
    load_dotenv()
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
            print(f"❌ {var} - NOT SET")
        else:
            print(f"✅ {var} - SET")
    
    if missing_vars:
        print(f"\n🚨 MISSING ENVIRONMENT VARIABLES: {missing_vars}")
        return False
    
    # Test 3: Test Config class import and validation
    try:
        from utils.config import Config
        print("✅ Config class imported successfully")
        
        # Test validation
        try:
            Config.validate_config()
            print("✅ Config validation passed")
            return True
        except Exception as e:
            print(f"❌ Config validation failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

if __name__ == "__main__":
    test_config()