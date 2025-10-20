import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, List, Tuple
import re
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedSentimentAnalyzer:
    """REWORKED Sentiment Analysis - Aggressive Non-Neutral Detection"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.positive_boost_words = {
            'excellent', 'amazing', 'awesome', 'great', 'good', 'fantastic', 'perfect',
            'love', 'wonderful', 'brilliant', 'outstanding', 'superb', 'terrific',
            'best', 'recommend', 'impressive', 'happy', 'pleased', 'satisfied'
        }
        self.negative_boost_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'disappointing',
            'poor', 'rubbish', 'garbage', 'trash', 'sucks', 'useless', 'broken',
            'waste', 'regret', 'avoid', 'disgusting', 'frustrating', 'annoying'
        }
        
        print(f"🔄 Loading sentiment model: {model_name}")
        
        # Device configuration
        device = 0 if torch.cuda.is_available() else -1
        print(f"🚀 Using {'GPU' if device == 0 else 'CPU'} for processing")
        
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                return_all_scores=True,
                device=device,
                max_length=512,
                truncation=True
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def calculate_text_polarity(self, text: str) -> float:
        """Calculate custom polarity score based on emotional words and patterns"""
        if not text or len(text) < 10:
            return 0.0
            
        text_lower = text.lower()
        
        # Count positive and negative indicators
        positive_count = sum(1 for word in self.positive_boost_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_boost_words if word in text_lower)
        
        # Punctuation intensity
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Capitalization intensity (shouting)
        words = text.split()
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
        
        # Calculate polarity score (-1 to +1)
        total_indicators = positive_count + negative_count + 0.1  # avoid division by zero
        
        if total_indicators == 0:
            return 0.0
            
        raw_polarity = (positive_count - negative_count) / total_indicators
        
        # Boost with punctuation and caps
        punctuation_boost = min(exclamation_count * 0.1, 0.3)
        caps_boost = min(caps_words * 0.05, 0.2)
        
        # Apply boosts to non-neutral texts
        if abs(raw_polarity) > 0.1:
            boost = punctuation_boost + caps_boost
            if raw_polarity > 0:
                final_polarity = raw_polarity + boost
            else:
                final_polarity = raw_polarity - boost
        else:
            final_polarity = raw_polarity
            
        return max(-1.0, min(1.0, final_polarity))
    
    def analyze_sentiment_aggressive(self, text: str) -> Dict:
        """Aggressive sentiment analysis that favors non-neutral classifications"""
        if not text or len(text.strip()) < 5:
            return self._create_sentiment_result('neutral', 0.0, 0.33, 0.34, 0.33)
        
        # Calculate custom polarity first
        polarity = self.calculate_text_polarity(text)
        
        try:
            # Get model prediction
            results = self.sentiment_pipeline([text])
            if not results:
                return self._fallback_based_on_polarity(polarity)
                
            sentiment_scores = results[0]
            sentiment_dict = {score['label']: score['score'] for score in sentiment_scores}
            
            # Get the model's primary sentiment
            model_sentiment = max(sentiment_dict.items(), key=lambda x: x[1])
            model_confidence = model_sentiment[1]
            
            # AGGRESSIVE OVERRIDE SYSTEM
            if abs(polarity) > 0.3:  # Strong custom polarity
                if polarity > 0.3 and sentiment_dict.get('positive', 0) > 0.2:
                    # Override to positive if custom polarity strongly positive
                    final_sentiment = 'positive'
                    confidence_boost = min(polarity * 0.3, 0.2)
                    final_confidence = sentiment_dict.get('positive', 0) + confidence_boost
                elif polarity < -0.3 and sentiment_dict.get('negative', 0) > 0.2:
                    # Override to negative if custom polarity strongly negative
                    final_sentiment = 'negative'
                    confidence_boost = min(abs(polarity) * 0.3, 0.2)
                    final_confidence = sentiment_dict.get('negative', 0) + confidence_boost
                else:
                    # Use model prediction but with polarity influence
                    final_sentiment = model_sentiment[0]
                    final_confidence = model_confidence
            else:
                # Weak polarity, use model but be more decisive
                if model_confidence < 0.6 and model_sentiment[0] == 'neutral':
                    # If model is unsure and says neutral, check if we should pick a side
                    if sentiment_dict.get('positive', 0) > sentiment_dict.get('negative', 0):
                        final_sentiment = 'positive'
                        final_confidence = sentiment_dict.get('positive', 0)
                    else:
                        final_sentiment = 'negative' 
                        final_confidence = sentiment_dict.get('negative', 0)
                else:
                    final_sentiment = model_sentiment[0]
                    final_confidence = model_confidence
            
            # Ensure confidence is bounded
            final_confidence = max(0.1, min(0.99, final_confidence))
            
            return {
                'sentiment': final_sentiment,
                'confidence': final_confidence,
                'negative_score': sentiment_dict.get('negative', 0),
                'neutral_score': sentiment_dict.get('neutral', 0),
                'positive_score': sentiment_dict.get('positive', 0),
                'custom_polarity': polarity
            }
            
        except Exception as e:
            print(f"❌ Error in aggressive analysis: {e}")
            return self._fallback_based_on_polarity(polarity)
    
    def _fallback_based_on_polarity(self, polarity: float) -> Dict:
        """Fallback sentiment based on custom polarity"""
        if polarity > 0.2:
            return self._create_sentiment_result('positive', 0.6 + polarity * 0.3, 0.2, 0.2, 0.6)
        elif polarity < -0.2:
            return self._create_sentiment_result('negative', 0.6 + abs(polarity) * 0.3, 0.6, 0.2, 0.2)
        else:
            return self._create_sentiment_result('neutral', 0.5, 0.33, 0.34, 0.33)
    
    def _create_sentiment_result(self, sentiment: str, confidence: float, neg: float, neut: float, pos: float) -> Dict:
        """Create a standardized sentiment result"""
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'negative_score': neg,
            'neutral_score': neut,
            'positive_score': pos
        }
    
    def analyze_sentences_aggressive(self, text: str) -> Dict:
        """Sentence-level analysis with aggressive non-neutral detection"""
        # Simple sentence splitting
        sentences = []
        for sentence in text.split('.'):
            sentence = sentence.strip()
            if sentence and len(sentence) > 15:  # Only meaningful sentences
                sentences.append(sentence)
        
        if not sentences:
            return self.analyze_sentiment_aggressive(text)
        
        sentence_sentiments = []
        
        for sentence in sentences:
            if 15 < len(sentence) < 400:  # Reasonable length
                sentiment_result = self.analyze_sentiment_aggressive(sentence)
                # Only consider confident or strongly polarized sentences
                if (sentiment_result['confidence'] > 0.6 or 
                    abs(sentiment_result.get('custom_polarity', 0)) > 0.4):
                    sentence_sentiments.append(sentiment_result)
        
        if not sentence_sentiments:
            return self.analyze_sentiment_aggressive(text)
        
        # Count sentiments
        sentiment_counts = Counter([s['sentiment'] for s in sentence_sentiments])
        
        # Prefer non-neutral if there's any strong signal
        if sentiment_counts.get('positive', 0) > 0 or sentiment_counts.get('negative', 0) > 0:
            if sentiment_counts.get('positive', 0) >= sentiment_counts.get('negative', 0):
                strongest_positive = max([s for s in sentence_sentiments if s['sentiment'] == 'positive'], 
                                       key=lambda x: x['confidence'], default=None)
                if strongest_positive:
                    return strongest_positive
            else:
                strongest_negative = max([s for s in sentence_sentiments if s['sentiment'] == 'negative'], 
                                       key=lambda x: x['confidence'], default=None)
                if strongest_negative:
                    return strongest_negative
        
        # Fallback to strongest overall
        return max(sentence_sentiments, key=lambda x: x['confidence'])
    
    def analyze_text_final(self, text: str) -> Dict:
        """Final text analysis with aggressive non-neutral preference"""
        if not text or len(text.strip()) < 10:
            return self._create_sentiment_result('neutral', 0.3, 0.33, 0.34, 0.33)
        
        # For short texts, use direct aggressive analysis
        if len(text) < 200:
            return self.analyze_sentiment_aggressive(text)
        # For long texts, use sentence-level aggressive analysis
        else:
            return self.analyze_sentences_aggressive(text)
    
    def get_detailed_sentiment(self, sentiment_scores: List) -> Dict:
        """Compatibility method"""
        if not sentiment_scores:
            return self._create_sentiment_result('neutral', 0.0, 0.0, 1.0, 0.0)
        
        try:
            sentiment_dict = {score['label']: score['score'] for score in sentiment_scores}
            primary = max(sentiment_dict.items(), key=lambda x: x[1])
            
            return {
                'sentiment': primary[0],
                'confidence': primary[1],
                'negative_score': sentiment_dict.get('negative', 0),
                'neutral_score': sentiment_dict.get('neutral', 0),
                'positive_score': sentiment_dict.get('positive', 0),
            }
        except:
            return self._create_sentiment_result('neutral', 0.0, 0.0, 1.0, 0.0)
    
    def calculate_weighted_sentiment(self, df: pd.DataFrame) -> Dict:
        """Calculate engagement-weighted sentiment"""
        if df.empty or 'engagement_score' not in df.columns:
            return {
                'weighted_positive': 0.33, 'weighted_negative': 0.33, 
                'weighted_neutral': 0.34, 'overall_sentiment': 'neutral'
            }
        
        try:
            total_engagement = df['engagement_score'].sum()
            if total_engagement == 0:
                return {
                    'weighted_positive': 0.33, 'weighted_negative': 0.33, 
                    'weighted_neutral': 0.34, 'overall_sentiment': 'neutral'
                }
            
            weighted_positive = (df[df['sentiment'] == 'positive']['engagement_score'].sum() / total_engagement 
                               if 'positive' in df['sentiment'].values else 0)
            weighted_negative = (df[df['sentiment'] == 'negative']['engagement_score'].sum() / total_engagement 
                               if 'negative' in df['sentiment'].values else 0)
            weighted_neutral = (df[df['sentiment'] == 'neutral']['engagement_score'].sum() / total_engagement 
                              if 'neutral' in df['sentiment'].values else 0)
            
            # Normalize
            total = weighted_positive + weighted_negative + weighted_neutral
            if total > 0:
                weighted_positive /= total
                weighted_negative /= total
                weighted_neutral /= total
            
            overall = max(['positive', 'negative', 'neutral'], 
                         key=lambda x: [weighted_positive, weighted_negative, weighted_neutral][['positive', 'negative', 'neutral'].index(x)])
            
            return {
                'weighted_positive': weighted_positive,
                'weighted_negative': weighted_negative,
                'weighted_neutral': weighted_neutral,
                'overall_sentiment': overall
            }
        except Exception as e:
            print(f"❌ Error in weighted sentiment: {e}")
            return {
                'weighted_positive': 0.33, 'weighted_negative': 0.33, 
                'weighted_neutral': 0.34, 'overall_sentiment': 'neutral'
            }
    
    def generate_sentiment_summary(self, df: pd.DataFrame) -> Dict:
        """Generate sentiment summary"""
        if df.empty:
            return self._create_empty_summary()
        
        try:
            total_posts = len(df)
            sentiment_counts = df['sentiment'].value_counts()
            sentiment_dist = {sentiment: count/total_posts for sentiment, count in sentiment_counts.items()}
            
            # Ensure all sentiments are present
            for sentiment in ['positive', 'neutral', 'negative']:
                if sentiment not in sentiment_dist:
                    sentiment_dist[sentiment] = 0.0
            
            weighted = self.calculate_weighted_sentiment(df)
            avg_confidence = df['confidence'].mean() if 'confidence' in df.columns else 0.5
            
            # Platform analysis
            platform_sentiment = {}
            if 'platform' in df.columns:
                try:
                    platform_sentiment = df.groupby('platform')['sentiment'].apply(
                        lambda x: x.value_counts(normalize=True).to_dict()
                    ).to_dict()
                except:
                    platform_sentiment = {}
            
            dominant = max(sentiment_dist.items(), key=lambda x: x[1])[0]
            
            return {
                'total_posts': total_posts,
                'sentiment_distribution': sentiment_dist,
                'weighted_sentiment': weighted,
                'average_confidence': avg_confidence,
                'platform_analysis': platform_sentiment,
                'dominant_sentiment': dominant
            }
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return self._create_empty_summary()
    
    def _create_empty_summary(self) -> Dict:
        """Create empty summary"""
        return {
            'total_posts': 0,
            'sentiment_distribution': {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0},
            'weighted_sentiment': {
                'weighted_positive': 0.33, 'weighted_negative': 0.33, 
                'weighted_neutral': 0.34, 'overall_sentiment': 'neutral'
            },
            'average_confidence': 0.0,
            'platform_analysis': {},
            'dominant_sentiment': 'neutral'
        }
    
    def create_sentiment_report(self, summary: Dict, brand_name: str) -> str:
        """Create sentiment report"""
        dominant = summary.get('dominant_sentiment', 'neutral')
        weighted = summary.get('weighted_sentiment', {}).get('overall_sentiment', 'neutral')
        
        emojis = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
        
        report = f"""
# 🎯 Sentiment Analysis Report for {brand_name}

## 📊 Quick Summary
- **Overall Sentiment**: {dominant.title()} {emojis.get(dominant, '')}
- **Engagement-Weighted**: {weighted.title()} {emojis.get(weighted, '')}
- **Total Posts Analyzed**: {summary.get('total_posts', 0):,}
- **Average Confidence**: {summary.get('average_confidence', 0):.1%}

## 📈 Detailed Breakdown
"""
        
        sentiment_dist = summary.get('sentiment_distribution', {})
        if sentiment_dist:
            report += "**Sentiment Distribution:**\n"
            for sentiment in ['positive', 'neutral', 'negative']:
                pct = sentiment_dist.get(sentiment, 0)
                report += f"- {sentiment.title()}: {pct:.1%}\n"
        
        weighted_sent = summary.get('weighted_sentiment', {})
        if weighted_sent:
            report += "\n**Weighted by Engagement:**\n"
            report += f"- Positive: {weighted_sent.get('weighted_positive', 0):.1%}\n"
            report += f"- Neutral: {weighted_sent.get('weighted_neutral', 0):.1%}\n"
            report += f"- Negative: {weighted_sent.get('weighted_negative', 0):.1%}\n"
        
        return report
    
    def analyze_complete(self, processed_data: pd.DataFrame, brand_name: str) -> Dict:
        """Complete analysis pipeline with aggressive non-neutral approach"""
        print("🧠 Starting AGGRESSIVE sentiment analysis...")
        
        if processed_data.empty:
            print("⚠️ No data to analyze!")
            empty_summary = self._create_empty_summary()
            return {
                'analyzed_data': pd.DataFrame(),
                'summary': empty_summary,
                'report': self.create_sentiment_report(empty_summary, brand_name),
                'sentiment_details': []
            }
        
        print(f"🔍 Analyzing {len(processed_data)} posts with aggressive non-neutral detection...")
        
        texts = processed_data['lemmatized_text'].fillna('').tolist()
        sentiment_details = []
        
        for i, text in enumerate(texts):
            if i % 25 == 0:
                print(f"   ...processed {i}/{len(texts)} posts")
            
            sentiment_detail = self.analyze_text_final(text)
            sentiment_details.append(sentiment_detail)
        
        # Add sentiment columns
        analyzed_data = processed_data.copy()
        sentiment_columns = ['sentiment', 'confidence', 'negative_score', 'neutral_score', 'positive_score']
        
        for col in sentiment_columns:
            analyzed_data[col] = [detail.get(col, 'neutral' if col == 'sentiment' else 0.0) 
                                for detail in sentiment_details]
        
        # Show distribution
        sentiment_counts = analyzed_data['sentiment'].value_counts()
        print(f"🎯 FINAL SENTIMENT DISTRIBUTION: {dict(sentiment_counts)}")
        
        summary = self.generate_sentiment_summary(analyzed_data)
        report = self.create_sentiment_report(summary, brand_name)
        
        return {
            'analyzed_data': analyzed_data,
            'summary': summary,
            'report': report,
            'sentiment_details': sentiment_details
        }