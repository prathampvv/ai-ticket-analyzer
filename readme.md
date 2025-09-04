# 🧠 AI-Powered Support Ticket Analyzer

This project analyzes customer support emails using AI + ML to extract insights like sentiment, priority, clustering, and customer risk scoring. It also generates business recommendations and visualizations automatically.

## 🚀 Features

1. Data Loading: Reads support emails from a CSV file (68b1acd44f393_Sample_Support_Emails_Dataset.csv).
2. Sentiment Analysis: Detects positive, neutral, and negative tickets.
3. Priority Classification: Identifies Critical, Urgent, and Normal tickets.
4. Topic Clustering: Groups similar tickets using TF-IDF + KMeans.
5. Customer Analysis: Finds high-risk customers with risk scoring.
6. Similarity Detection: Detects duplicate/similar tickets for automation.
7. Business Insights: Auto-generates insights and recommendations.
8. Visualizations: Saves charts to ai_support_analysis.png (also pops up automatically).
9. Outputs Saved:
    analyzed_tickets.csv → detailed ticket analysis
    customer_analysis.csv → customer insights
    ai_support_analysis.png → visualizations

## 🛠 Installation
1. Create and activate a virtual environment (recommended):
    python -m venv venv
    source venv/bin/activate     # On Mac/Linux
    venv\Scripts\activate        # On Windows
2. Install all dependencies:
    pip install -r requirements.txt

## 📂File Structure:
### 📁 project-root
1. │── main.py                     # Main script (run this file)
2. │── requirements.txt             # Python dependencies
3. │── README.md                    # Project documentation
4. │── 68b1acd44f393_Sample_Support_Emails_Dataset.csv   # Input dataset
5. │── analyzed_tickets.csv         # (Generated) Ticket analysis
6. │── customer_analysis.csv        # (Generated) Customer insights
7. │── ai_support_analysis.png      # (Generated) Visualization charts

## ▶️ Running the Project:
1. Place your dataset CSV file in the project folder.:
    (Default: 68b1acd44f393_Sample_Support_Emails_Dataset.csv)
2. Run the analysis with:
    python main.py
3. Outputs generated:
    analyzed_tickets.csv 
    customer_analysis.csv 
    ai_support_analysis.png

## 🤖 AI Techniques Used:
1. Natural Language Processing (NLP) → TF-IDF, Sentiment (TextBlob)
2. Machine Learning → KMeans clustering, similarity detection
3. Predictive Insights → Risk scoring, escalation suggestions
4. Visualization → Matplotlib + Seaborn

## Example output:
[View Example Output](https://drive.google.com/file/d/1-UT5MPWwzQhzPbJ4gnDkW_971CTjtlWp/view?usp=sharing)

## 👨‍💻 Author

Pratham Verma
BTech CSE, VIT Vellore
