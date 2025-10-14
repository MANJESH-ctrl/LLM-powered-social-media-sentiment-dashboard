import pandas as pd
import re
import numpy as np
from typing import Dict, List, Tuple
from transformers import pipeline
import emoji
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class AdvancedDataProcessor:
    """Advanced data processing pipeline for social media sentiment analysis"""
    
    

    def __init__(self):
    # Download required NLTK data
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            pass
    
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) 

        self.text_cleaner = pipeline("text-classification", 
                                   model="cardiffnlp/twitter-roberta-base-offensive",
                                   tokenizer="cardiffnlp/twitter-roberta-base-offensive")
    
    def load_data(self, brand_name: str, platforms: List[str]) -> pd.DataFrame:
        """Load and combine data from multiple platforms"""
        all_data = []
        
        for platform in platforms:
            try:
                # Look for the latest data file
                import os
                data_dir = "outputs/raw_data"
                pattern = f"{brand_name}_{platform.lower()}_"
                files = [f for f in os.listdir(data_dir) 
                        if f.startswith(pattern) and f.endswith('.csv')]
                
                if files:
                    latest_file = sorted(files)[-1]
                    filepath = os.path.join(data_dir, latest_file)
                    df = pd.read_csv(filepath)
                    df['platform'] = platform.lower()
                    all_data.append(df)
            except Exception as e:
                print(f"Error loading {platform} data: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def advanced_text_cleaning(self, text: str) -> str:
        """Comprehensive text cleaning and preprocessing"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags but keep the text
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Handle emojis - convert to text description
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Remove special characters but keep basic punctuation for context
        text = re.sub(r'[^\w\s\.!?,]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def lemmatize_text(self, text: str) -> str:
        """Advanced text normalization using lemmatization"""
        tokens = word_tokenize(text)
        filtered_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and token.isalpha()
        ]
        return ' '.join(filtered_tokens)
    
    def calculate_engagement_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive engagement scores"""
        df_processed = df.copy()
        
        # Platform-specific engagement calculations
        if 'reddit' in df_processed['platform'].values:
            reddit_mask = df_processed['platform'] == 'reddit'
            reddit_data = df_processed[reddit_mask]
            
            # Normalize Reddit metrics
            if 'score' in reddit_data.columns:
                df_processed.loc[reddit_mask, 'normalized_score'] = (
                    reddit_data['score'] - reddit_data['score'].min()
                ) / (reddit_data['score'].max() - reddit_data['score'].min() + 1e-8)
            
            if 'num_comments' in reddit_data.columns:
                df_processed.loc[reddit_mask, 'comment_engagement'] = (
                    reddit_data['num_comments'] / (reddit_data['num_comments'].max() + 1e-8)
                )
            
            # Combined Reddit engagement score
            engagement_factors = []
            if 'normalized_score' in df_processed.columns:
                engagement_factors.append(df_processed.loc[reddit_mask, 'normalized_score'])
            if 'comment_engagement' in df_processed.columns:
                engagement_factors.append(df_processed.loc[reddit_mask, 'comment_engagement'])
            
            if engagement_factors:
                df_processed.loc[reddit_mask, 'engagement_score'] = np.mean(engagement_factors, axis=0)
        
        # YouTube engagement calculations
        if 'youtube' in df_processed['platform'].values:
            youtube_mask = df_processed['platform'] == 'youtube'
            youtube_data = df_processed[youtube_mask]
            
            if 'likes' in youtube_data.columns:
                df_processed.loc[youtube_mask, 'normalized_likes'] = (
                    youtube_data['likes'] - youtube_data['likes'].min()
                ) / (youtube_data['likes'].max() - youtube_data['likes'].min() + 1e-8)
            
            if 'comments_count' in youtube_data.columns:
                df_processed.loc[youtube_mask, 'comment_ratio'] = (
                    youtube_data['comments_count'] / (youtube_data['comments_count'].max() + 1e-8)
                )
            
            # Combined YouTube engagement score
            yt_engagement_factors = []
            if 'normalized_likes' in df_processed.columns:
                yt_engagement_factors.append(df_processed.loc[youtube_mask, 'normalized_likes'])
            if 'comment_ratio' in df_processed.columns:
                yt_engagement_factors.append(df_processed.loc[youtube_mask, 'comment_ratio'])
            
            if yt_engagement_factors:
                df_processed.loc[youtube_mask, 'engagement_score'] = np.mean(yt_engagement_factors, axis=0)
        
        # Fill NaN engagement scores
        df_processed['engagement_score'] = df_processed['engagement_score'].fillna(0.5)
        
        return df_processed
    
    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that will augment sentiment analysis"""
        df_features = df.copy()
        
        # Text length features
        df_features['text_length'] = df_features['text'].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
        df_features['word_count'] = df_features['text'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        # Sentiment intensity indicators
        df_features['exclamation_count'] = df_features['text'].apply(
            lambda x: str(x).count('!') if pd.notna(x) else 0
        )
        df_features['question_count'] = df_features['text'].apply(
            lambda x: str(x).count('?') if pd.notna(x) else 0
        )
        
        # Capitalization ratio (often indicates strong emotion)
        df_features['caps_ratio'] = df_features['text'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1) 
            if pd.notna(x) else 0
        )
        
        return df_features
    
    def process_pipeline(self, brand_name: str, platforms: List[str]) -> pd.DataFrame:
        """Complete data processing pipeline"""
        print("🚀 Starting advanced data processing...")
        
        # Load data
        raw_data = self.load_data(brand_name, platforms)
        if raw_data.empty:
            raise ValueError("No data found for the specified brand and platforms")
        
        print(f"📊 Loaded {len(raw_data)} records from {platforms}")
        
        # Clean text
        print("🧹 Cleaning text data...")
        raw_data['cleaned_text'] = raw_data['text'].apply(self.advanced_text_cleaning)
        
        # Lemmatize
        print("🔤 Lemmatizing text...")
        raw_data['lemmatized_text'] = raw_data['cleaned_text'].apply(self.lemmatize_text)
        
        # Calculate engagement metrics
        print("📈 Calculating engagement metrics...")
        processed_data = self.calculate_engagement_metrics(raw_data)
        
        # Create additional features
        print("🎨 Creating sentiment features...")
        final_data = self.create_sentiment_features(processed_data)
        
        print("✅ Data processing completed successfully!")
        return final_data