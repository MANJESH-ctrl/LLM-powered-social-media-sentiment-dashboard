from src.utils.config import Config

def main():
    """Main entry point for the application."""
    try:
        Config.validate_config()
        print("✅ Configuration validated successfully!")
        print("🚀 Starting Streamlit dashboard...")
        print("📊 Open your browser and navigate to the localhost address shown below")
        
        # Import and run streamlit app
        from src.dashboard.app12 import main as st_main
        st_main()
        
    except Exception as e:
        print(f"❌ Application failed to start: {str(e)}")
        print("💡 Please check your .env file and API credentials")

if __name__ == "__main__":
    main()