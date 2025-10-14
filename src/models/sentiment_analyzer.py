import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, List, Tuple
from collections import Counter

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis with comprehensive metrics - FIXED VERSION"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        
        print(f"🔄 Loading model: {model_name}")
        
        # FIX: Simplified pipeline setup to avoid device mapping issues
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            return_all_scores=True,
            device=-1  # Force CPU to avoid device mapping issues
        )
        
        self.sentiment_labels = ["negative", "neutral", "positive"]
    
    def analyze_sentiment_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for batch of texts - SIMPLIFIED"""
        if not texts:
            return []
        
        results = []
        batch_size = 8  # Smaller batch size for stability
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                # Filter out empty texts
                valid_texts = [text for text in batch_texts if text and len(text.strip()) > 0]
                if not valid_texts:
                    continue
                    
                batch_results = self.sentiment_pipeline(valid_texts)
                results.extend(batch_results)
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # Fallback: neutral sentiment for failed analyses
                fallback_result = [[
                    {"label": "neutral", "score": 1.0},
                    {"label": "negative", "score": 0.0},
                    {"label": "positive", "score": 0.0}
                ]] * len(batch_texts)
                results.extend(fallback_result)
        
        return results
    
    def get_detailed_sentiment(self, sentiment_scores: List) -> Dict:
        """Extract detailed sentiment information"""
        if not sentiment_scores:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'negative_score': 0.0,
                'neutral_score': 1.0,
                'positive_score': 0.0,
            }
        
        sentiment_dict = {score['label']: score['score'] for score in sentiment_scores}
        
        # Determine primary sentiment
        primary_sentiment = max(sentiment_dict.items(), key=lambda x: x[1])
        
        return {
            'sentiment': primary_sentiment[0],
            'confidence': primary_sentiment[1],
            'negative_score': sentiment_dict.get('negative', 0),
            'neutral_score': sentiment_dict.get('neutral', 0),
            'positive_score': sentiment_dict.get('positive', 0),
        }
    
    def calculate_weighted_sentiment(self, df: pd.DataFrame) -> Dict:
        """Calculate engagement-weighted sentiment metrics"""
        if 'engagement_score' not in df.columns or df['engagement_score'].sum() == 0:
            return {
                'weighted_positive': 0.33,
                'weighted_negative': 0.33,
                'weighted_neutral': 0.34,
                'overall_sentiment': 'neutral'
            }
        
        total_engagement = df['engagement_score'].sum()
        
        weighted_positive = (
            df[df['sentiment'] == 'positive']['engagement_score'].sum() / total_engagement
        )
        weighted_negative = (
            df[df['sentiment'] == 'negative']['engagement_score'].sum() / total_engagement
        )
        weighted_neutral = (
            df[df['sentiment'] == 'neutral']['engagement_score'].sum() / total_engagement
        )
        
        # Determine overall sentiment
        sentiments = {
            'positive': weighted_positive,
            'negative': weighted_negative,
            'neutral': weighted_neutral
        }
        overall_sentiment = max(sentiments.items(), key=lambda x: x[1])[0]
        
        return {
            'weighted_positive': weighted_positive,
            'weighted_negative': weighted_negative,
            'weighted_neutral': weighted_neutral,
            'overall_sentiment': overall_sentiment
        }
    
    def generate_sentiment_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive sentiment summary"""
        total_posts = len(df)
        
        if total_posts == 0:
            return {
                'total_posts': 0,
                'sentiment_distribution': {},
                'weighted_sentiment': {
                    'weighted_positive': 0, 'weighted_negative': 0, 
                    'weighted_neutral': 0, 'overall_sentiment': 'neutral'
                },
                'average_confidence': 0,
                'platform_analysis': {},
                'high_engagement_sentiment': {},
                'low_engagement_sentiment': {},
                'dominant_sentiment': 'neutral'
            }
        
        # Basic sentiment counts
        sentiment_counts = df['sentiment'].value_counts()
        sentiment_percentages = {
            sentiment: count / total_posts 
            for sentiment, count in sentiment_counts.items()
        }
        
        # Weighted sentiment
        weighted_sentiment = self.calculate_weighted_sentiment(df)
        
        # Confidence metrics
        avg_confidence = df['confidence'].mean() if 'confidence' in df.columns else 0.5
        
        # Platform comparison
        platform_sentiment = {}
        if 'platform' in df.columns:
            platform_sentiment = df.groupby('platform')['sentiment'].apply(
                lambda x: x.value_counts(normalize=True).to_dict()
            ).to_dict()
        
        return {
            'total_posts': total_posts,
            'sentiment_distribution': sentiment_percentages,
            'weighted_sentiment': weighted_sentiment,
            'average_confidence': avg_confidence,
            'platform_analysis': platform_sentiment,
            'dominant_sentiment': max(sentiment_percentages.items(), key=lambda x: x[1])[0] if sentiment_percentages else 'neutral'
        }
    
    def create_sentiment_report(self, summary: Dict, brand_name: str) -> str:
        """Generate human-readable sentiment report"""
        sentiment_emojis = {
            'positive': '😊',
            'negative': '😞',
            'neutral': '😐'
        }
        
        dominant = summary.get('dominant_sentiment', 'neutral')
        weighted = summary.get('weighted_sentiment', {}).get('overall_sentiment', 'neutral')
        
        report = f"""
# 🎯 Sentiment Analysis Report for {brand_name}

## 📊 Quick Summary
- **Overall Sentiment**: {dominant.title()} {sentiment_emojis.get(dominant, '')}
- **Engagement-Weighted Sentiment**: {weighted.title()} {sentiment_emojis.get(weighted, '')}
- **Total Posts Analyzed**: {summary.get('total_posts', 0):,}
- **Average Confidence**: {summary.get('average_confidence', 0):.1%}

## 📈 Detailed Breakdown
"""
        
        # Add sentiment distribution if available
        sentiment_dist = summary.get('sentiment_distribution', {})
        if sentiment_dist:
            report += "**Sentiment Distribution:**\n"
            for sentiment in ['positive', 'neutral', 'negative']:
                pct = sentiment_dist.get(sentiment, 0)
                report += f"- {sentiment.title()}: {pct:.1%}\n"
        
        # Add weighted sentiment if available
        weighted = summary.get('weighted_sentiment', {})
        if weighted:
            report += "\n**Weighted by Engagement:**\n"
            report += f"- Positive: {weighted.get('weighted_positive', 0):.1%}\n"
            report += f"- Neutral: {weighted.get('weighted_neutral', 0):.1%}\n"
            report += f"- Negative: {weighted.get('weighted_negative', 0):.1%}\n"
        
        return report
    
    def analyze_complete(self, processed_data: pd.DataFrame, brand_name: str) -> Dict:
        """Complete sentiment analysis pipeline - SIMPLIFIED"""
        print("🧠 Starting advanced sentiment analysis...")
        
        # Analyze sentiment for all texts
        texts = processed_data['lemmatized_text'].fillna('').tolist()
        
        # Filter out very short texts
        valid_texts = [text for text in texts if len(str(text).strip()) > 10]
        print(f"📝 Analyzing {len(valid_texts)} valid texts...")
        
        sentiment_results = self.analyze_sentiment_batch(valid_texts)
        
        # Add sentiment columns to dataframe
        sentiment_details = [self.get_detailed_sentiment(result) for result in sentiment_results]
        
        # Ensure we have the same number of results as rows
        for i, detail in enumerate(sentiment_details):
            if i < len(processed_data):
                for key, value in detail.items():
                    processed_data.loc[processed_data.index[i], key] = value
        
        # Fill any missing sentiment values
        processed_data['sentiment'] = processed_data['sentiment'].fillna('neutral')
        processed_data['confidence'] = processed_data['confidence'].fillna(0.5)
        
        # Generate comprehensive summary
        summary = self.generate_sentiment_summary(processed_data)
        report = self.create_sentiment_report(summary, brand_name)
        
        return {
            'analyzed_data': processed_data,
            'summary': summary,
            'report': report,
            'sentiment_details': sentiment_details
        }



