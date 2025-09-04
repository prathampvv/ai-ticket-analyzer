# 🧠 AI-Powered Support Ticket Analyzer

This project analyzes customer support emails using AI + ML to extract insights like sentiment, priority, clustering, and customer risk scoring. It also generates business recommendations and visualizations automatically.

## 🚀 Features

Data Loading: Reads support emails from a CSV file (68b1acd44f393_Sample_Support_Emails_Dataset.csv).
Sentiment Analysis: Detects positive, neutral, and negative tickets.
Priority Classification: Identifies Critical, Urgent, and Normal tickets.
Topic Clustering: Groups similar tickets using TF-IDF + KMeans.
Customer Analysis: Finds high-risk customers with risk scoring.
Similarity Detection: Detects duplicate/similar tickets for automation.
Business Insights: Auto-generates insights and recommendations.
Visualizations: Saves charts to ai_support_analysis.png (also pops up automatically).
Outputs Saved:
    analyzed_tickets.csv → detailed ticket analysis
    customer_analysis.csv → customer insights
    ai_support_analysis.png → visualizations

## 🛠 Installation
Create and activate a virtual environment (recommended):
    python -m venv venv
    source venv/bin/activate     # On Mac/Linux
    venv\Scripts\activate        # On Windows
Install all dependencies:
    pip install -r requirements.txt

## 📂File Structure:
📁 project-root
│── main.py                     # Main script (run this file)
│── requirements.txt             # Python dependencies
│── README.md                    # Project documentation
│── 68b1acd44f393_Sample_Support_Emails_Dataset.csv   # Input dataset
│── analyzed_tickets.csv         # (Generated) Ticket analysis
│── customer_analysis.csv        # (Generated) Customer insights
│── ai_support_analysis.png      # (Generated) Visualization charts

## ▶️ Running the Project:
Place your dataset CSV file in the project folder.:
    (Default: 68b1acd44f393_Sample_Support_Emails_Dataset.csv)
Run the analysis with:
    python main.py
Outputs generated:
    analyzed_tickets.csv 
    customer_analysis.csv 
    ai_support_analysis.png

## 🤖 AI Techniques Used:
    Natural Language Processing (NLP) → TF-IDF, Sentiment (TextBlob)
    Machine Learning → KMeans clustering, similarity detection
    Predictive Insights → Risk scoring, escalation suggestions
    Visualization → Matplotlib + Seaborn

## 👨‍💻 Author

Pratham Verma
BTech CSE, VIT Vellore
