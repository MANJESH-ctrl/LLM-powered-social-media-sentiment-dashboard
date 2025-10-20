import streamlit as st
import sys
import os
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go

# Multiple path approaches to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..')
project_root = os.path.join(current_dir, '..', '..')

# Try multiple path approaches
sys.path.insert(0, src_path)
sys.path.insert(0, project_root)
sys.path.insert(0, os.getcwd())

# Import with fallback options
Config = None
try:
    from utils.config import Config
    st.success("✅ Config import successful (direct)")
except ImportError:
    try:
        from src.utils.config import Config  
        st.success("✅ Config import successful (with src)")
    except ImportError as e:
        st.error(f"❌ All Config imports failed: {e}")

# Import other modules
try:
    from data.collectors.reddit_collector import RedditCollector
    from data.collectors.youtube_collector import YouTubeCollector
    from data.storage.csv_storage import CSVStorage
    from data.processing.data_processor import AdvancedDataProcessor
    from models.sentiment_analyzer import AdvancedSentimentAnalyzer
    st.success("✅ All other imports successful!")
except ImportError as e:
    st.error(f"❌ Other import error: {e}")

def setup_page():
    """Configure Streamlit page settings with professional styling"""
    st.set_page_config(
        page_title="Brand Sentiment Intelligence",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional look
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .platform-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .strategy-info {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎯 Brand Sentiment Intelligence</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced AI-Powered Social Media Sentiment Analysis")

def show_analysis_sidebar() -> dict:
    """Render sidebar with improved keyword guidance"""
    st.sidebar.header("🔍 Analysis Configuration")
    
    # Brand name (required)
    brand_name = st.sidebar.text_input(
        "Brand/Product Name", 
        "iPhone",
        help="Enter the main brand or product you want to analyze"
    )
    
    # Keywords (optional, with better explanation)
    keywords = st.sidebar.text_input(
        "Additional Topics (optional)", 
        "battery, camera, performance",
        help="Optional: Add specific topics separated by commas. We'll try these first, then use brand-only if needed."
    )
    
    st.sidebar.markdown("💡 **Smart Search Examples:**")
    st.sidebar.markdown("- `iPhone` + `battery, camera`")
    st.sidebar.markdown("- `Tesla` + `autopilot, charging`") 
    st.sidebar.markdown("- `Nike` + `comfort, durability`")
    st.sidebar.markdown("- *Leave empty for brand-only search*")
    
    # Platform selection
    platforms = st.sidebar.multiselect(
        "Select Platforms",
        ["Reddit", "YouTube"],
        default=["Reddit"],
        help="Choose which social media platforms to analyze"
    )
    
    # Time period
    time_period = st.sidebar.selectbox(
        "Time Period",
        ["all", "year", "month", "week", "day"],
        index=0,
        help="Select how far back to collect data from"
    )
    
    # Analysis limit
    analysis_limit = st.sidebar.slider(
        "Posts per Platform",
        min_value=10,
        max_value=100,
        value=30,
        help="Number of posts to collect and analyze from each platform"
    )
    
    return {
        'brand_name': brand_name.strip(),
        'keywords': keywords.strip(),
        'platforms': platforms,
        'time_period': time_period,
        'analysis_limit': analysis_limit
    }

def validate_inputs(params: dict) -> tuple:
    """Validate user inputs before analysis"""
    if not params['brand_name']:
        return False, "Please enter a brand name"
    
    if not params['platforms']:
        return False, "Please select at least one platform"
    
    # Validate keywords (max 5)
    keywords_list = [kw.strip() for kw in params['keywords'].split(',') if kw.strip()]
    if len(keywords_list) > 5:
        return False, "Please enter maximum 5 keywords"
    
    return True, "All inputs valid"

def collect_data(params: dict, storage: CSVStorage) -> tuple:
    """Collect data from selected platforms with enhanced feedback"""
    success_count = 0
    collection_details = {}
    
    for platform in params['platforms']:
        with st.spinner(f"🔍 Collecting data from {platform}..."):
            try:
                # Initialize appropriate collector
                if platform.lower() == "reddit":
                    collector = RedditCollector()
                else:  # YouTube
                    collector = YouTubeCollector()
                
                # Collect data using improved strategy system
                data = collector.collect_data(
                    brand_name=params['brand_name'],
                    keywords=params['keywords'],
                    time_period=params['time_period'],
                    limit=params['analysis_limit']
                )
                
                if data and len(data) > 0:
                    # Save to CSV
                    filepath = storage.save_data(
                        data, 
                        params['brand_name'], 
                        platform.lower()
                    )
                    
                    if filepath:
                        success_count += 1
                        
                        # Collect strategy information
                        strategies_used = list(set([item.get('search_strategy', 'Unknown') for item in data]))
                        collection_details[platform] = {
                            'count': len(data),
                            'strategies': strategies_used,
                            'sample_titles': [item.get('title', 'No title')[:50] + '...' for item in data[:3]]
                        }
                        
                        st.success(f"✅ Collected {len(data)} items from {platform}")
                        
                        # Show strategy info in expander
                        with st.expander(f"🔍 View {platform} collection details"):
                            st.markdown(f"**Strategies used:**")
                            for strategy in strategies_used:
                                st.markdown(f"- `{strategy}`")
                            
                            st.markdown(f"**Sample content:**")
                            for i, title in enumerate(collection_details[platform]['sample_titles']):
                                st.markdown(f"{i+1}. {title}")
                        
                    else:
                        st.warning(f"⚠️ Data collected from {platform} but failed to save")
                else:
                    # More helpful guidance
                    st.warning(f"🔍 No data found for '{params['brand_name']}' on {platform}")
                    st.info("""
                    **Tips to get better results:**
                    - Try a more popular brand name
                    - Use simpler, more common keywords  
                    - Try a longer time period ('all' or 'year')
                    - Some brands aren't heavily discussed on all platforms
                    """)
                    
            except Exception as e:
                st.error(f"❌ Error collecting from {platform}: {str(e)}")
    
    return success_count > 0, collection_details

def display_sentiment_overview(summary: dict):
    """Display comprehensive sentiment overview with metrics"""
    st.header("📊 Sentiment Overview")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Posts", f"{summary.get('total_posts', 0):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        dominant = summary.get('dominant_sentiment', 'neutral')
        emoji_map = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Dominant Sentiment", f"{dominant.title()} {emoji_map.get(dominant, '')}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Confidence", f"{summary.get('average_confidence', 0):.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        weighted = summary.get('weighted_sentiment', {}).get('overall_sentiment', 'neutral')
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Weighted Sentiment", f"{weighted.title()} {emoji_map.get(weighted, '')}")
        st.markdown('</div>', unsafe_allow_html=True)

def create_sentiment_visualizations(analyzed_data: pd.DataFrame, summary: dict):
    """Create interactive sentiment visualizations"""
    st.header("📈 Sentiment Analytics")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution pie chart
        sentiment_counts = analyzed_data['sentiment'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.3,
            marker_colors=['#2ecc71', '#f39c12', '#e74c3c']  # green, orange, red
        )])
        fig_pie.update_layout(
            title_text="📊 Sentiment Distribution",
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Platform comparison bar chart
        platform_sentiment = analyzed_data.groupby(['platform', 'sentiment']).size().reset_index(name='count')
        fig_bar = px.bar(
            platform_sentiment,
            x='platform',
            y='count',
            color='sentiment',
            title="📱 Sentiment by Platform",
            color_discrete_map={
                'positive': '#2ecc71',
                'neutral': '#f39c12', 
                'negative': '#e74c3c'
            },
            height=400
        )
        fig_bar.update_layout(
            xaxis_title="Platform",
            yaxis_title="Number of Posts"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Engagement vs Sentiment scatter plot
    st.subheader("Engagement vs Sentiment Analysis")
    if 'engagement_score' in analyzed_data.columns and 'confidence' in analyzed_data.columns:
        fig_scatter = px.scatter(
            analyzed_data,
            x='engagement_score',
            y='confidence',
            color='sentiment',
            size='word_count' if 'word_count' in analyzed_data.columns else None,
            hover_data=['title'],
            title="🎯 Engagement Score vs Sentiment Confidence",
            color_discrete_map={
                'positive': '#2ecc71',
                'neutral': '#f39c12',
                'negative': '#e74c3c'
            },
            height=500
        )
        fig_scatter.update_layout(
            xaxis_title="Engagement Score",
            yaxis_title="Sentiment Confidence"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

def display_detailed_insights(analysis_results: dict, params: dict, collection_details: dict):
    """Display detailed analysis insights and data"""
    analyzed_data = analysis_results['analyzed_data']
    summary = analysis_results['summary']
    report = analysis_results['report']
    
    # Display collection strategy summary
    if collection_details:
        st.header("🔍 Collection Strategy Summary")
        for platform, details in collection_details.items():
            with st.expander(f"{platform} Collection Details"):
                st.markdown(f"**Items collected:** {details['count']}")
                st.markdown("**Search strategies used:**")
                for strategy in details['strategies']:
                    st.markdown(f"- `{strategy}`")
    
    # Display the AI-generated report
    st.header("📋 AI Analysis Report")
    st.markdown(report)
    
    # Display visualizations
    create_sentiment_visualizations(analyzed_data, summary)
    
    # Platform-specific insights
    st.header("🔍 Platform Insights")
    platform_cols = st.columns(len(params['platforms']))
    
    for idx, platform in enumerate(params['platforms']):
        with platform_cols[idx]:
            platform_data = analyzed_data[analyzed_data['platform'] == platform.lower()]
            if not platform_data.empty:
                platform_sentiment = platform_data['sentiment'].value_counts(normalize=True)
                
                st.markdown(f'<div class="platform-card">', unsafe_allow_html=True)
                st.subheader(f"{platform} Analysis")
                st.metric("Total Posts", len(platform_data))
                
                for sentiment, percentage in platform_sentiment.items():
                    st.write(f"- {sentiment.title()}: {percentage:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample analyzed data
    st.header("📝 Sample Analyzed Content")
    display_columns = ['platform', 'title', 'sentiment', 'confidence', 'engagement_score']
    available_columns = [col for col in display_columns if col in analyzed_data.columns]
    
    if available_columns:
        st.dataframe(
            analyzed_data[available_columns].head(10),
            use_container_width=True,
            height=400
        )
    
    # Download functionality
    st.header("💾 Export Results")
    csv_data = analyzed_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Complete Analysis (CSV)",
        data=csv_data,
        file_name=f"{params['brand_name']}_sentiment_analysis.csv",
        mime="text/csv",
        use_container_width=True
    )

def run_complete_analysis(params: dict):
    """Run the complete analysis pipeline"""
    try:
        # Initialize components
        storage = CSVStorage()
        processor = AdvancedDataProcessor()
        analyzer = AdvancedSentimentAnalyzer()
        
        # Step 1: Data Collection
        st.header("🚀 Step 1: Data Collection")
        collection_success, collection_details = collect_data(params, storage)
        
        if not collection_success:
            st.error("❌ Data collection failed. Please check your API credentials and try again.")
            return
        
        # Step 2: Data Processing
        st.header("🔄 Step 2: Data Processing")
        with st.spinner("Processing and cleaning data for analysis..."):
            processed_data = processor.process_pipeline(
                params['brand_name'], 
                [p.lower() for p in params['platforms']]
            )
        
        if processed_data.empty:
            st.error("❌ No data available for analysis after processing. Please try different search parameters.")
            return
        
        st.success(f"✅ Successfully processed {len(processed_data)} records")
        
        # Step 3: Sentiment Analysis
        st.header("🧠 Step 3: AI Sentiment Analysis")
        with st.spinner("Analyzing sentiment with advanced AI model... This may take a few minutes."):
            analysis_results = analyzer.analyze_complete(processed_data, params['brand_name'])
        
        # Step 4: Display Results
        st.header("📊 Step 4: Analysis Results")
        display_sentiment_overview(analysis_results['summary'])
        display_detailed_insights(analysis_results, params, collection_details)
        
        st.balloons()
        st.success("🎉 Analysis completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Analysis pipeline failed: {str(e)}")
        st.info("💡 Please check your API credentials and try again with different parameters")

def main():
    """Main application function"""
    setup_page()
    
    # Check if Config was imported successfully
    if 'Config' not in globals() or Config is None:
        st.error("❌ Configuration system not available")
        st.info("""
        💡 Please ensure:
        1. Your .env file exists in the project root
        2. All required environment variables are set
        3. The project structure is correct
        """)
        return
    
    # Configuration validation
    try:
        Config.validate_config()
        st.success("✅ Configuration validated successfully!")
    except ValueError as e:
        st.error(f"❌ Configuration Error: {str(e)}")
        st.info("""
        💡 Please check your `.env` file and ensure all API credentials are set:
        - REDDIT_CLIENT_ID
        - REDDIT_CLIENT_SECRET  
        - YOUTUBE_API_KEY
        """)
        return
    
    # Get analysis parameters from sidebar
    params = show_analysis_sidebar()
    
    # Main analysis section
    st.header("🎯 Start Brand Sentiment Analysis")
    st.markdown("""
    Configure your analysis parameters in the sidebar and click **Analyze Brand Sentiment** 
    to generate comprehensive AI-powered insights about your brand's social media presence.
    
    ### 🆕 Smart Keyword System
    - **Multiple Strategies**: We automatically try different search approaches
    - **Never Fails**: If your keywords don't work, we use brand-only search
    - **Emotional Focus**: Always targets opinionated content for better sentiment analysis
    """)
    
    # Analysis button
    if st.button("🚀 Analyze Brand Sentiment", type="primary", use_container_width=True):
        # Validate inputs
        is_valid, message = validate_inputs(params)
        
        if not is_valid:
            st.warning(f"⚠️ {message}")
            return
        
        # Run complete analysis
        run_complete_analysis(params)
    
    # Instructions and information
    with st.expander("📖 How to Use This Dashboard"):
        st.markdown("""
        ### 🎯 Complete Sentiment Analysis Workflow
        
        **1. Configure Parameters:**
           - **Brand Name**: The brand you want to analyze
           - **Keywords**: Optional specific topics (we'll try multiple strategies)
           - **Platforms**: Choose Reddit, YouTube, or both
           - **Time Period**: How far back to collect data
           - **Posts per Platform**: Number of posts to analyze
        
        **2. Smart Collection Process:**
           - **Strategy 1**: Brand + your keywords + emotional indicators
           - **Strategy 2**: Brand + emotional indicators only  
           - **Strategy 3**: Brand only (never fails!)
           - We automatically use the strategy that works best
        
        **3. Advanced Processing:**
           - Emotional content filtering
           - Text cleaning and normalization
           - Engagement scoring
           - AI sentiment analysis
        
        **4. Comprehensive Results:**
           - Overall sentiment metrics
           - Platform-specific insights
           - Interactive visualizations
           - Strategy transparency
           - Downloadable reports
        
        ### 🔧 Technical Features
        - **Multi-Strategy Collection**: Never shows "no data" errors
        - **Emotional Filtering**: Better sentiment analysis results
        - **Platform Optimization**: Different search strategies per platform
        - **Progress Tracking**: See which strategies worked
        - **Error Resilience**: Continues despite individual failures
        """)

if __name__ == "__main__":
    main()