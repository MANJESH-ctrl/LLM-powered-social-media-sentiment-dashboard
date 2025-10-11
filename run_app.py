# run_app.py - Simple entry point
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.dashboard.app import main

if __name__ == "__main__":
    main()
