# Social Media Sentiment Analysis using Machine Learning

A machine learning-based Natural Language Processing (NLP) project that classifies social media text into **positive, negative, and neutral sentiments**. This project builds an end-to-end text classification pipeline including preprocessing, feature extraction, model training, and performance evaluation.

---

## Project Overview

Understanding public sentiment on social media is crucial for businesses, policymakers, and researchers. This project leverages NLP techniques and machine learning algorithms to automatically analyze textual data and determine the underlying sentiment.

### Workflow Includes:
- Data cleaning and preprocessing  
- Tokenization and stopword removal  
- Stemming and lemmatization  
- Exploratory Data Analysis (EDA)  
- TF-IDF feature extraction  
- Model training and evaluation  
- Comparative performance analysis  

---

## Dataset

The dataset contains labeled social media sentences categorized into:

✅ Positive  
✅ Negative  
✅ Neutral  

*Note: The dataset is not included in this repository due to size and data-sharing restrictions.*

---

## Models Implemented

- Logistic Regression  
- Naive Bayes  
- Random Forest  

These models were selected for their effectiveness in text classification tasks.

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|------------|-------------|----------|------------|
| Logistic Regression | **72.11%** | 70.05% | **72.11%** | **68.95%** |
| Naive Bayes | 65.95% | **69.87%** | 65.95% | 59.44% |
| Random Forest | 65.35% | 63.57% | 65.35% | 63.20% |

---

## Best Performing Model

**Logistic Regression** achieved the highest overall accuracy and F1 score, making it the most effective model for this sentiment classification task.

> The results highlight the strength of linear models when working with TF-IDF features in NLP problems.

---

## Key Insight

> Simpler models like Logistic Regression outperformed more complex algorithms, demonstrating that well-engineered text features can significantly impact model performance.

---

## Tech Stack

### **Languages & Libraries**
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- NLTK  
- Matplotlib  
- Seaborn  
- WordCloud  

---

## How to Run This Project

### Clone the repository
```bash
git clone https://github.com/Ar9ab007cpu/social-media-sentiment-analysis-ml.git
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Add the dataset  
Place the dataset file inside the project directory.

### Run the notebook
```bash
jupyter notebook
```

Open:

```
Sentiment_Analysis.ipynb
```

---

## Project Highlights

✅ Built a complete NLP pipeline from preprocessing to evaluation  
✅ Implemented TF-IDF for feature extraction  
✅ Compared multiple machine learning algorithms  
✅ Performed multi-class sentiment classification  
✅ Derived insights through model comparison  

---

## Future Improvements

- Hyperparameter tuning  
- Implement deep learning models (LSTM / BERT)  
- Deploy as a web application  
- Real-time sentiment analysis  
- Expand dataset for improved accuracy  

---

## Support

If you found this project useful, consider **starring ⭐ the repository** — it helps others discover it!

---


