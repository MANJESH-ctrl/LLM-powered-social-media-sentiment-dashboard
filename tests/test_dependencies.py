import sys
import subprocess
import importlib

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("🔧 TESTING DEPENDENCIES...")
    
    required_packages = [
        'streamlit', 'praw', 'googleapiclient', 'pandas', 
        'transformers', 'torch', 'plotly', 'nltk', 'emoji',
        'sklearn', 'dotenv', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Handle special cases for package names
            if package == 'googleapiclient':
                import googleapiclient.discovery
            elif package == 'sklearn':
                import sklearn
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n🚨 MISSING PACKAGES: {missing_packages}")
        print("💡 Install with: uv add " + " ".join(missing_packages))
        return False
    else:
        print("🎉 All dependencies are installed!")
        return True

if __name__ == "__main__":
    test_dependencies()