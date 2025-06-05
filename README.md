# 🍽Zomato Restaurant Analysis & Recommendation System

An end-to-end Data Science project analyzing Zomato restaurant data using Python, Power BI, and Streamlit. This project includes data cleaning, exploratory data analysis (EDA), clustering, and a restaurant recommender system.

---

## Live App

[Try the Streamlit App](https://zomato-data-analysis-k4quwpmyjtdmtymag7cxkd.streamlit.app)

---

## Project Structure

├── CLUSTERING/                 # Clustering logic + report
│   ├── clustering.py
│   └── Clustering Analysis Report.pdf

├── EDA/                        # Exploratory data analysis
│   ├── eda.py
│   └── Exploratory Data Analysis Report.docx

├── DATASETS/
│   └── zomato_cleaned.csv
│   └── BangaloreZomatoData.csv
│   └── zomato_cluster.csv

├── DASHBOARD.pbix             # Power BI insights dashboard

├── restaurant_recommender.py  # Streamlit recommender app
├── data_cleaning.py           # Data cleaning script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation


---

##  Key Features

### Exploratory Data Analysis
- Top cuisines and popular areas
- Cost vs. rating insights
- Missing value handling and feature engineering

### Clustering
- KMeans clustering on restaurant profiles
- Segmenting restaurants based on ratings and cost

### Power BI Dashboard
- Top cuisines by rating
- Popular restaurant areas
- Price vs. rating scatterplot
- Cluster breakdown

### Recommender System
- Content-based filtering
- Filter by cuisine, restaurant, and get suggestions instantly
- Built and deployed using Streamlit

---

## How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
streamlit run restaurant_recommender.py
