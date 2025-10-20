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
    

    def calculate_emotional_intensity(self, text: str) -> float:
        """More sensitive emotional intensity scoring"""
        if not isinstance(text, str):
            return 0.0

        # Expanded emotional indicators
        intensity_indicators = {
            # Strong positive
            'love': 2.0, 'amazing': 1.8, 'awesome': 1.8, 'excellent': 1.7, 'fantastic': 1.7,
            'perfect': 1.6, 'brilliant': 1.6, 'outstanding': 1.6, 'superb': 1.5, 'terrific': 1.5,
            'best': 1.4, 'great': 1.3, 'good': 1.0, 'recommend': 1.2, 'impressive': 1.2,
            'happy': 1.3, 'pleased': 1.2, 'satisfied': 1.2, 'wonderful': 1.5,
        
            # Strong negative  
            'hate': 2.0, 'terrible': 1.8, 'awful': 1.8, 'horrible': 1.8, 'worst': 1.7,
            'disappointing': 1.6, 'rubbish': 1.6, 'garbage': 1.6, 'trash': 1.5, 'sucks': 1.7,
            'useless': 1.4, 'broken': 1.3, 'waste': 1.4, 'regret': 1.5, 'avoid': 1.3,
            'disgusting': 1.7, 'frustrating': 1.4, 'annoying': 1.3, 'poor': 1.2, 'bad': 1.1,
        
            # Moderate indicators
            'like': 0.8, 'dislike': 0.8, 'okay': 0.3, 'decent': 0.5, 'fine': 0.3,
            'issue': 0.6, 'problem': 0.7, 'bug': 0.6, 'crash': 0.8, 'slow': 0.7,
            'fast': 0.6, 'smooth': 0.5, 'easy': 0.4, 'difficult': 0.6,
        
            # Punctuation and patterns
            '!': 0.4, '?': 0.2, '!!': 0.8, '!!!': 1.2,
        
            # Emojis
            '😍': 1.8, '😊': 1.2, '👍': 1.0, '❤️': 1.5, '🔥': 1.3,
            '😠': 1.5, '😡': 1.8, '👎': 1.0, '💔': 1.5, '😞': 1.3,
            '😂': 1.0, '🎉': 1.2, '💩': 1.6
        }

        score = 0.0
        text_lower = text.lower()

        for indicator, weight in intensity_indicators.items():
            if indicator in ['!', '?', '!!', '!!!']:
                # Count punctuation
                count = text.count(indicator)
                score += count * weight
            elif len(indicator) == 1:  # Single character emojis
                count = text.count(indicator)
                score += count * weight
            else:
                # Count word occurrences (including partial matches for emphasis)
                count = text_lower.count(indicator.lower())
                score += count * weight

        # Boost for ALL CAPS words (shouting)
        caps_words = [word for word in text.split() if word.isupper() and len(word) > 2]
        score += len(caps_words) * 0.8

        # Boost for repeated punctuation patterns
        repeated_exclamations = len(re.findall(r'!{2,}', text))
        score += repeated_exclamations * 0.5

        return min(10.0, score)  # Cap at reasonable maximum
    


    
    

    def filter_emotional_content(self, df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
        """Keep only texts with emotional intensity above threshold"""
        print("🎭 Calculating emotional intensity for all posts...")
        df['emotional_intensity'] = df['text'].apply(self.calculate_emotional_intensity)
    
        emotional_data = df[df['emotional_intensity'] >= threshold]
        print(f"🎭 Filtered to {len(emotional_data)} emotional posts (intensity ≥ {threshold}) out of {len(df)} total")
    
        # Show intensity distribution
        intensity_stats = df['emotional_intensity'].describe()
        print(f"📊 Emotional intensity stats - Min: {intensity_stats['min']:.2f}, Max: {intensity_stats['max']:.2f}, Mean: {intensity_stats['mean']:.2f}")
    
        return emotional_data



    def process_pipeline(self, brand_name: str, platforms: List[str]) -> pd.DataFrame:
        print("🚀 Starting advanced data processing...")
    
        # Load data
        raw_data = self.load_data(brand_name, platforms)
        if raw_data.empty:
            raise ValueError("No data found for the specified brand and platforms")
    
        print(f"📊 Loaded {len(raw_data)} records from {platforms}")
    
        # LOWERED THRESHOLD: More sensitive emotional filtering
        print("🎭 Filtering for emotional content with LOWERED threshold...")
        emotional_data = self.filter_emotional_content(raw_data, threshold=0.3)  # Changed from 0.8 to 0.3
    
        if emotional_data.empty:
            print("⚠️ No emotional content found after filtering! Using all data.")
            emotional_data = raw_data
    
        # Rest of processing remains the same...
        print("🧹 Cleaning text data...")
        emotional_data['cleaned_text'] = emotional_data['text'].apply(self.advanced_text_cleaning)
    
        print("🔤 Lemmatizing text...")
        emotional_data['lemmatized_text'] = emotional_data['cleaned_text'].apply(self.lemmatize_text)
    
        print("📈 Calculating engagement metrics...")
        processed_data = self.calculate_engagement_metrics(emotional_data)
    
        print("🎨 Creating sentiment features...")
        final_data = self.create_sentiment_features(processed_data)
    
        print("✅ Data processing completed successfully!")
        return final_data    
    

   