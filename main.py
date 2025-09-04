import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AITicketAnalyzer:
    def __init__(self):
        """Initialize the AI-powered ticket analyzer"""
        self.tickets_df = None
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.kmeans = KMeans(n_clusters=5, random_state=42)

    def load_data(self, file_path):
        """Load and preprocess ticket data from CSV"""
        self.tickets_df = pd.read_csv(file_path)

        # Ensure datetime format
        self.tickets_df['sent_date'] = pd.to_datetime(self.tickets_df['sent_date'], errors='coerce')

        # Combine subject + body for NLP
        self.tickets_df['combined_text'] = (
            self.tickets_df['subject'].fillna('') + ' ' + self.tickets_df['body'].fillna('')
        )

        print("✅ Data loaded successfully!")
        print(f"📊 Total tickets: {len(self.tickets_df)}")
        return self.tickets_df

    def sentiment_analysis(self):
        """Perform sentiment analysis on tickets"""
        print("\n🧠 Performing AI Sentiment Analysis...")

        sentiments = []
        polarities = []

        for text in self.tickets_df['combined_text']:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            polarities.append(polarity)

            if polarity > 0.1:
                sentiments.append('Positive')
            elif polarity < -0.1:
                sentiments.append('Negative')
            else:
                sentiments.append('Neutral')

        self.tickets_df['sentiment'] = sentiments
        self.tickets_df['polarity'] = polarities

        sentiment_counts = self.tickets_df['sentiment'].value_counts()
        print(f"Sentiment Distribution: {dict(sentiment_counts)}")

        return sentiment_counts

    def priority_classification(self):
        """AI-based priority classification"""
        print("\n🎯 AI Priority Classification...")

        priorities = []
        for _, row in self.tickets_df.iterrows():
            text = str(row['combined_text']).lower()
            subject = str(row['subject']).lower()

            critical_keywords = ['critical', 'urgent', 'down', 'server', 'inaccessible', 'immediately', 'emergency']
            urgent_keywords = ['urgent', 'blocked', 'cannot', 'unable', 'error', 'billing', 'charged']

            if any(keyword in text for keyword in critical_keywords) or any(keyword in subject for keyword in critical_keywords):
                priorities.append('Critical')
            elif any(keyword in text for keyword in urgent_keywords) or any(keyword in subject for keyword in urgent_keywords):
                priorities.append('Urgent')
            else:
                priorities.append('Normal')

        self.tickets_df['priority'] = priorities
        priority_counts = self.tickets_df['priority'].value_counts()
        print(f"Priority Distribution: {dict(priority_counts)}")

        return priority_counts

    def topic_clustering(self):
        """ML-based topic clustering"""
        print("\n🤖 ML Topic Clustering...")

        tfidf_matrix = self.vectorizer.fit_transform(self.tickets_df['combined_text'].fillna(''))
        clusters = self.kmeans.fit_predict(tfidf_matrix)
        self.tickets_df['topic_cluster'] = clusters

        feature_names = self.vectorizer.get_feature_names_out()
        cluster_centers = self.kmeans.cluster_centers_

        topics = {}
        for i, center in enumerate(cluster_centers):
            top_indices = center.argsort()[-5:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            topics[f'Cluster_{i}'] = top_terms
            print(f"Topic {i}: {', '.join(top_terms)}")

        return topics

    def customer_analysis(self):
        """Analyze customer patterns"""
        print("\n👥 Customer Behavior Analysis...")

        customer_stats = self.tickets_df.groupby('sender').agg({
            'subject': 'count',
            'priority': lambda x: (x == 'Critical').sum(),
            'sentiment': lambda x: (x == 'Negative').sum(),
            'polarity': 'mean'
        }).round(3)

        customer_stats.columns = ['total_tickets', 'critical_tickets', 'negative_tickets', 'avg_sentiment']
        customer_stats = customer_stats.sort_values('total_tickets', ascending=False)

        customer_stats['risk_score'] = (
            customer_stats['critical_tickets'] * 0.4 +
            customer_stats['negative_tickets'] * 0.3 +
            (customer_stats['total_tickets'] / customer_stats['total_tickets'].max()) * 0.3
        ).round(2)

        print("🔥 High-Risk Customers (Top 3):")
        high_risk = customer_stats.sort_values('risk_score', ascending=False).head(3)
        for customer, data in high_risk.iterrows():
            print(f"  {customer}: Risk Score {data['risk_score']} | {data['total_tickets']} tickets | {data['critical_tickets']} critical")

        return customer_stats

    def similarity_detection(self):
        """Detect similar tickets"""
        print("\n🔍 Detecting Similar Tickets...")

        tfidf_matrix = self.vectorizer.fit_transform(self.tickets_df['combined_text'].fillna(''))
        similarity_matrix = cosine_similarity(tfidf_matrix)

        similar_pairs = []
        threshold = 0.3

        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > threshold:
                    similar_pairs.append((i, j, similarity_matrix[i][j]))

        print(f"Found {len(similar_pairs)} similar ticket pairs (similarity > {threshold})")

        return similar_pairs

    def generate_insights(self):
        """Generate AI-powered business insights"""
        print("\n💡 AI-Generated Business Insights:")

        insights = []

        top_customer = self.tickets_df['sender'].value_counts().index[0]
        top_count = self.tickets_df['sender'].value_counts().iloc[0]
        insights.append(f"🎯 Customer Focus: {top_customer} represents {(top_count/len(self.tickets_df)*100):.1f}% of tickets")

        critical_pct = (self.tickets_df['priority'] == 'Critical').sum() / len(self.tickets_df) * 100
        insights.append(f"⚠️ Critical Issues: {critical_pct:.1f}% of tickets are critical")

        negative_pct = (self.tickets_df['sentiment'] == 'Negative').sum() / len(self.tickets_df) * 100
        insights.append(f"😔 Customer Satisfaction: {negative_pct:.1f}% negative sentiment")

        for insight in insights:
            print(f"  {insight}")

        return insights

    def create_visualizations(self):
        """Create AI analysis visualizations"""
        print("\n📊 Creating AI Analysis Visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AI-Powered Support Ticket Analysis', fontsize=16, fontweight='bold')

        priority_counts = self.tickets_df['priority'].value_counts()
        axes[0, 0].pie(priority_counts.values, labels=priority_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Priority Distribution')

        sentiment_counts = self.tickets_df['sentiment'].value_counts()
        colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
        axes[0, 1].bar(sentiment_counts.index, sentiment_counts.values,
                       color=[colors[sent] for sent in sentiment_counts.index])
        axes[0, 1].set_title('Sentiment Analysis')

        customer_counts = self.tickets_df['sender'].value_counts().head(5)
        axes[0, 2].barh(range(len(customer_counts)), customer_counts.values)
        axes[0, 2].set_yticks(range(len(customer_counts)))
        axes[0, 2].set_yticklabels([email.split('@')[0] for email in customer_counts.index])
        axes[0, 2].set_title('Top 5 Customers by Ticket Volume')

        cluster_counts = self.tickets_df['topic_cluster'].value_counts().sort_index()
        axes[1, 0].bar(cluster_counts.index, cluster_counts.values, color='skyblue')
        axes[1, 0].set_title('ML Topic Clusters')

        priority_sentiment = pd.crosstab(self.tickets_df['priority'], self.tickets_df['sentiment'])
        sns.heatmap(priority_sentiment, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Priority vs Sentiment Matrix')

        daily_tickets = self.tickets_df.groupby(self.tickets_df['sent_date'].dt.date).size()
        axes[1, 2].plot(daily_tickets.index, daily_tickets.values, marker='o', linewidth=2)
        axes[1, 2].set_title('Daily Ticket Volume')

        plt.tight_layout()
        plt.savefig('ai_support_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📈 Visualizations saved as 'ai_support_analysis.png' and displayed on screen")


    def ai_recommendations(self):
        """Generate AI implementation recommendations based on actual data"""
        print("\n🚀 AI Implementation Recommendations:")

        recommendations = []

        if (self.tickets_df['priority'] == 'Critical').sum() > 0:
            recommendations.append("1. 🤖 Implement NLP-based Auto-Classification:")
            recommendations.append("   - Use TF-IDF + SVM for priority classification")
            recommendations.append("   - Deploy BERT for sentiment & urgency detection")

        login_tickets = self.tickets_df['combined_text'].str.contains("login|password", case=False).sum()
        if login_tickets > 0:
            recommendations.append("\n2. 🎯 Smart Ticket Routing System:")
            recommendations.append(f"   - Handle {login_tickets} password/login tickets automatically")
            recommendations.append("   - Escalate server/critical issues directly to Tier 3")

        if not recommendations:
            recommendations.append("✅ Current system is stable — no urgent AI improvements needed.")

        for rec in recommendations:
            print(rec)

        return recommendations

    def run_complete_analysis(self, file_path):
        """Run the complete AI analysis pipeline"""
        print("🚀 Starting AI-Powered Support Ticket Analysis")
        print("=" * 60)

        self.load_data(file_path)
        self.sentiment_analysis()
        self.priority_classification()
        self.topic_clustering()
        customer_stats = self.customer_analysis()
        self.similarity_detection()
        self.generate_insights()
        self.create_visualizations()
        self.ai_recommendations()

        self.tickets_df.to_csv('analyzed_tickets.csv', index=False)
        customer_stats.to_csv('customer_analysis.csv')

        print("\n✅ Analysis Complete!")
        print("📁 Results saved to:")
        print("   - analyzed_tickets.csv")
        print("   - customer_analysis.csv")
        print("   - ai_support_analysis.png")

        return self.tickets_df, customer_stats


if __name__ == "__main__":
    analyzer = AITicketAnalyzer()
    tickets_df, customer_stats = analyzer.run_complete_analysis("68b1acd44f393_Sample_Support_Emails_Dataset.csv")

    print("\n📋 EXECUTIVE SUMMARY:")
    print(f"📊 Total Tickets Analyzed: {len(tickets_df)}")
    print(f"👥 Unique Customers: {tickets_df['sender'].nunique()}")
    print(f"⚠️ Critical Tickets: {(tickets_df['priority'] == 'Critical').sum()}")
    print(f"😔 Negative Sentiment: {(tickets_df['sentiment'] == 'Negative').sum()}")
    print(f"🎯 Top Customer: {tickets_df['sender'].value_counts().index[0]} "
          f"({tickets_df['sender'].value_counts().iloc[0]} tickets)")
