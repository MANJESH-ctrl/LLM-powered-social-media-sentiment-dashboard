import pandas as pd
import re
import os   
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
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"⚠️ NLTK download warning: {e}")

class AdvancedDataProcessor:
    """Advanced data processing pipeline for social media sentiment analysis"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) 
        # self.text_cleaner = pipeline("text-classification", 
        #                            model="cardiffnlp/twitter-roberta-base-offensive",
        #                            tokenizer="cardiffnlp/twitter-roberta-base-offensive")
    

    def load_data(self, brand_name: str, platforms: List[str]) -> pd.DataFrame:
        all_data = []
        data_dir = "outputs/raw_data"  # Consider making this configurable
    
        for platform in platforms:
            try:
                pattern = f"{brand_name}_{platform.lower()}_"
                files = [f for f in os.listdir(data_dir) 
                        if f.startswith(pattern) and f.endswith('.csv')]
            
                if files:
                    latest_file = sorted(files)[-1]
                    filepath = os.path.join(data_dir, latest_file)
                    df = pd.read_csv(filepath)
                    df['platform'] = platform.lower()
                    all_data.append(df)
                    print(f"📂 Loaded {len(df)} records from {latest_file}")
                else:
                    print(f"📭 No data files found for {brand_name} on {platform}")
                    
            except Exception as e:
                print(f"❌ Error loading {platform} data: {e}")
    
        if not all_data:
            print(f"🚫 No data loaded for {brand_name} on any platform")
            return pd.DataFrame()
    
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"📊 Combined {len(combined_data)} total records")
        return combined_data
    


    def calculate_emotional_intensity(self, text: str) -> float:
        """More sensitive emotional intensity scoring with word boundary precision"""
        if not isinstance(text, str) or not text.strip():
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
                # 🚀 FIXED: Use word boundaries to count only whole words
                pattern = r'\b' + re.escape(indicator) + r'\b'
                count = len(re.findall(pattern, text_lower))
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

        # If text becomes too short after cleaning, return empty
        if len(text) < 3:
            return ""
        
        return text
    
    
    def lemmatize_text(self, text: str) -> str:
        """Advanced text normalization using lemmatization"""
        if not text or len(text.strip()) < 3:
            return ""
        try:
            tokens = word_tokenize(text)
            filtered_tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token not in self.stop_words and token.isalpha()
            ]
            # return ' '.join(filtered_tokens)
            return ' '.join(filtered_tokens) if filtered_tokens else ""
        except Exception as e:
            print(f"⚠️ Lemmatization error: {e}")
            return text  # Fallback to original text
        

         
    def calculate_engagement_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive engagement scores with robust normalization"""
        df_processed = df.copy()
    
        def safe_normalize(series):
            """Safe min-max normalization handling edge cases"""
            if series.empty or series.nunique() <= 1:
                # Return neutral 0.5 for all if no variance or empty
                return pd.Series([0.5] * len(series), index=series.index)
            return (series - series.min()) / (series.max() - series.min() + 1e-8)  # Avoid division by zero
    
        # Reddit engagement calculations
        if 'reddit' in df_processed['platform'].values:
            reddit_mask = df_processed['platform'] == 'reddit'
            reddit_data = df_processed[reddit_mask]
         
            engagement_factors = []
        
            # Normalize score (upvotes - downvotes)
            if 'score' in reddit_data.columns:
                df_processed.loc[reddit_mask, 'normalized_score'] = safe_normalize(reddit_data['score'])
                engagement_factors.append(df_processed.loc[reddit_mask, 'normalized_score'])
        
            # Normalize comment engagement
            if 'num_comments' in reddit_data.columns:
                df_processed.loc[reddit_mask, 'comment_engagement'] = safe_normalize(reddit_data['num_comments'])
                engagement_factors.append(df_processed.loc[reddit_mask, 'comment_engagement'])
        
            # Calculate combined engagement score
            if engagement_factors:
                df_processed.loc[reddit_mask, 'engagement_score'] = np.mean(engagement_factors, axis=0)
    
        # YouTube engagement calculations
        if 'youtube' in df_processed['platform'].values:
            youtube_mask = df_processed['platform'] == 'youtube'
            youtube_data = df_processed[youtube_mask]
        
            yt_engagement_factors = []
        
            # Normalize likes for videos and comments
            if 'likes' in youtube_data.columns:
                df_processed.loc[youtube_mask, 'normalized_likes'] = safe_normalize(youtube_data['likes'])
                yt_engagement_factors.append(df_processed.loc[youtube_mask, 'normalized_likes'])
        
            # Normalize comment ratio for videos
            if 'comments_count' in youtube_data.columns:
                df_processed.loc[youtube_mask, 'comment_ratio'] = safe_normalize(youtube_data['comments_count'])
                yt_engagement_factors.append(df_processed.loc[youtube_mask, 'comment_ratio'])
        
            # Calculate combined YouTube engagement score
            if yt_engagement_factors:
                df_processed.loc[youtube_mask, 'engagement_score'] = np.mean(yt_engagement_factors, axis=0)
    
        # Fill NaN engagement scores with platform-specific averages
        platform_means = df_processed.groupby('platform')['engagement_score'].transform('mean')
        df_processed['engagement_score'] = df_processed['engagement_score'].fillna(platform_means)
    
        # Final fallback for any remaining NaN values
        df_processed['engagement_score'] = df_processed['engagement_score'].fillna(0.5)
    
        # Ensure engagement scores are between 0 and 1
        df_processed['engagement_score'] = df_processed['engagement_score'].clip(0, 1)
    
        print(f"📊 Engagement scores calculated: {df_processed['engagement_score'].describe()['mean']:.2f} average")
        return df_processed     
    
    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that will augment sentiment analysis with error handling"""
        df_features = df.copy()
    
        try:
            # Text length features
            df_features['text_length'] = df_features['text'].apply(
                lambda x: len(str(x)) if pd.notna(x) and isinstance(x, str) else 0
           )
            # Sentiment intensity indicators
            df_features['exclamation_count'] = df_features['text'].apply(
                lambda x: str(x).count('!') if pd.notna(x) and isinstance(x, str) else 0
            )
            df_features['question_count'] = df_features['text'].apply(
                lambda x: str(x).count('?') if pd.notna(x) and isinstance(x, str) else 0
            )
        
            # Capitalization ratio (often indicates strong emotion)
            df_features['caps_ratio'] = df_features['text'].apply(
                lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1) 
                if pd.notna(x) and isinstance(x, str) and len(str(x)) > 0 else 0
            )
        
            # Sentiment word density (using cleaned text)
            df_features['sentiment_word_density'] = df_features['cleaned_text'].apply(
                lambda x: self.calculate_emotional_intensity(x) / max(len(str(x)), 1) 
                if pd.notna(x) and isinstance(x, str) and len(str(x)) > 0 else 0
            )
        
            print(f"🎨 Created {len([col for col in df_features.columns if 'feature' in col or 'count' in col or 'ratio' in col])} sentiment features")
        
        except Exception as e:
            print(f"⚠️ Error creating sentiment features: {e}")
            # Return original dataframe if feature creation fails
            return df
    
        return df_features
    
    

    def process_pipeline(self, brand_name: str, platforms: List[str]) -> pd.DataFrame:
        """Complete data processing pipeline with enhanced progress tracking and error handling"""
        print(f"🚀 Starting advanced data processing for '{brand_name}' on {platforms}")
    
        # Step 1: Load data from all platforms
        print("📥 Step 1: Loading data from storage...")
        raw_data = self.load_data(brand_name, platforms)
    
        if raw_data.empty:
            error_msg = f"❌ No data found for '{brand_name}' on platforms: {platforms}"
            print(error_msg)
            raise ValueError(error_msg)
    
        print(f"   Loaded {len(raw_data)} total records")
        platform_counts = raw_data['platform'].value_counts()
        for platform, count in platform_counts.items():
            print(f"   - {platform}: {count} records")
    
        # Step 2: Emotional content filtering
        print("\n🎭 Step 2: Filtering emotional content...")
        emotional_data = self.filter_emotional_content(raw_data, threshold=0.3)
    
        if emotional_data.empty:
            print("   ⚠️  No emotional content found after filtering! Using all data (expect mostly neutral results)")
            emotional_data = raw_data
        else:
            print(f"   ✅ Kept {len(emotional_data)} emotional posts ({len(emotional_data)/len(raw_data)*100:.1f}% of original)")
    
        # Step 3: Text cleaning
        print("\n🧹 Step 3: Cleaning text data...")
        initial_count = len(emotional_data)
        emotional_data['cleaned_text'] = emotional_data['text'].apply(self.advanced_text_cleaning)
    
         # Remove rows where cleaning resulted in empty or very short text
        emotional_data = emotional_data[emotional_data['cleaned_text'].str.len() > 3]
        cleaned_removed = initial_count - len(emotional_data)
        if cleaned_removed > 0:
            print(f"   Removed {cleaned_removed} posts with empty text after cleaning")
        print(f"   {len(emotional_data)} posts remaining after text cleaning")
    
        # Step 4: Text lemmatization
        print("\n🔤 Step 4: Lemmatizing text...")
        emotional_data['lemmatized_text'] = emotional_data['cleaned_text'].apply(self.lemmatize_text)
    
        # Remove rows where lemmatization resulted in empty text
        lemmatized_initial = len(emotional_data)
        emotional_data = emotional_data[emotional_data['lemmatized_text'].str.len() > 3]
        lemmatized_removed = lemmatized_initial - len(emotional_data)
        if lemmatized_removed > 0:
            print(f"   Removed {lemmatized_removed} posts with empty text after lemmatization")
        print(f"   {len(emotional_data)} posts remaining after lemmatization")
    
       # Step 5: Engagement scoring
        print("\n📈 Step 5: Calculating engagement metrics...")
        processed_data = self.calculate_engagement_metrics(emotional_data)
    
        # Step 6: Feature engineering
        print("\n🎨 Step 6: Creating sentiment features...")
        final_data = self.create_sentiment_features(processed_data)
    
        # Final summary
        print(f"\n✅ PROCESSING COMPLETE!")
        print(f"📊 Final dataset: {len(final_data)} records")
        print(f"🎯 Emotional posts: {len(final_data[final_data['emotional_intensity'] >= 0.3])} ({len(final_data[final_data['emotional_intensity'] >= 0.3])/len(final_data)*100:.1f}%)")
    
        # Platform breakdown
        if 'platform' in final_data.columns:
            platform_final = final_data['platform'].value_counts()
            print(f"🌐 Platform distribution:")
            for platform, count in platform_final.items():
                print(f"   - {platform}: {count} records")
        
        # Engagement overview
        if 'engagement_score' in final_data.columns:
            eng_stats = final_data['engagement_score'].describe()
            print(f"📈 Engagement score: avg {eng_stats['mean']:.2f} (min {eng_stats['min']:.2f}, max {eng_stats['max']:.2f})")
    
        return final_data



    

   