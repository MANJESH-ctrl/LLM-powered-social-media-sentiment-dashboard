# import streamlit as st
# import sys
# import os
# from datetime import datetime, timedelta

import streamlit as st
import sys
import os

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..')
sys.path.insert(0, src_path)

# Now import your modules
try:
    from data.collectors.reddit_collector import RedditCollector
    from data.collectors.youtube_collector import YouTubeCollector
    from data.storage.csv_storage import CSVStorage
    from utils.config import Config
except ImportError as e:
    st.error(f"Import Error: {e}")

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.collectors.reddit_collector import RedditCollector
from src.data.collectors.youtube_collector import YouTubeCollector
from src.data.storage.csv_storage import CSVStorage
from src.utils.config import Config

def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Social Media Sentiment Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Social Media Sentiment Analysis Dashboard")
    st.markdown("Collect and analyze social media data from Reddit and YouTube")

def show_sidebar() -> dict:
    """Render sidebar inputs and return parameters."""
    st.sidebar.header("🔍 Data Collection Parameters")
    
    brand_name = st.sidebar.text_input("Brand Name", "Apple")
    keywords = st.sidebar.text_input("Additional Keywords", "tech, innovation, product")
    
    platforms = st.sidebar.multiselect(
        "Platforms",
        ["Reddit", "YouTube"],
        default=["Reddit"]
    )
    
    time_period = st.sidebar.selectbox(
        "Time Period",
        ["all", "year", "month", "week", "day"],
        index=0
    )
    
    post_limit = st.sidebar.slider(
        "Maximum Posts/Videos to Collect",
        min_value=10,
        max_value=200,
        value=50
    )
    
    return {
        'brand_name': brand_name,
        'keywords': keywords,
        'platforms': platforms,
        'time_period': time_period,
        'post_limit': post_limit
    }

def collect_and_display_data(params: dict, storage: CSVStorage):
    """Handle data collection and display results."""
    if not params['platforms']:
        st.warning("Please select at least one platform")
        return
    
    if st.button("🚀 Collect Data", type="primary"):
        with st.spinner("Collecting data from selected platforms..."):
            for platform in params['platforms']:
                st.subheader(f"🔍 Collecting from {platform}")
                
                try:
                    if platform.lower() == "reddit":
                        collector = RedditCollector()
                    else:  # YouTube
                        collector = YouTubeCollector()
                    
                    # Collect data
                    data = collector.collect_data(
                        brand_name=params['brand_name'],
                        keywords=params['keywords'],
                        time_period=params['time_period'],
                        limit=params['post_limit']
                    )
                    
                    if data:
                        # Save to CSV
                        filepath = storage.save_data(
                            data, 
                            params['brand_name'], 
                            platform.lower()
                        )
                        
                        # Display success message
                        st.success(f"✅ Collected {len(data)} items from {platform}")
                        st.info(f"💾 Data saved to: `{filepath}`")
                        
                        # Show data preview
                        st.subheader(f"📋 Data Preview from {platform}")
                        preview_df = storage.read_latest_data(
                            params['brand_name'], 
                            platform.lower()
                        )
                        
                        if preview_df is not None:
                            st.dataframe(preview_df.head(10), use_container_width=True)
                            
                            # Show basic statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Items", len(preview_df))
                            with col2:
                                st.metric("Columns", len(preview_df.columns))
                            with col3:
                                st.metric("Data Types", preview_df['content_type'].nunique() if 'content_type' in preview_df.columns else "N/A")
                            
                            # Show sample of text data
                            st.subheader("📝 Sample Text Content")
                            if 'text' in preview_df.columns:
                                sample_texts = preview_df['text'].dropna().head(3).tolist()
                                for i, text in enumerate(sample_texts):
                                    with st.expander(f"Sample Text {i+1}"):
                                        st.write(text[:500] + "..." if len(text) > 500 else text)
                        else:
                            st.error("Failed to load collected data for preview")
                    
                    else:
                        st.warning(f"No data collected from {platform}. Try adjusting your search parameters.")
                        
                except Exception as e:
                    st.error(f"❌ Error collecting from {platform}: {str(e)}")
                    st.info("💡 Check your API credentials and internet connection")

def main():
    """Main application function."""
    setup_page()
    
    # Validate configuration
    try:
        Config.validate_config()
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.info("Please check your `.env` file and ensure all API credentials are set")
        return
    
    # Initialize storage
    storage = CSVStorage()
    
    # Show sidebar and get parameters
    params = show_sidebar()
    
    # Main content area
    st.header("🎯 Data Collection")
    st.markdown("Configure your parameters in the sidebar and click 'Collect Data' to start")
    
    # Collect and display data
    collect_and_display_data(params, storage)
    
    # Instructions section
    with st.expander("📖 Instructions"):
        st.markdown("""
        ### How to use this dashboard:
        
        1. **Set Parameters**: Use the sidebar to configure:
           - Brand name you want to analyze
           - Additional keywords (comma-separated)
           - Platforms to collect from
           - Time period for the data
           - Limit for number of posts/videos
        
        2. **Collect Data**: Click the "Collect Data" button to start data collection
        
        3. **Review Data**: Check the preview to understand your data structure
        
        4. **Next Steps**: Once satisfied with data collection, we'll proceed to sentiment analysis
        
        ### Required API Setup:
        - **Reddit**: Create a Reddit app at https://www.reddit.com/prefs/apps
        - **YouTube**: Get API key from https://console.cloud.google.com/
        """)

if __name__ == "__main__":
    main()