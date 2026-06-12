# Project Summary

This notebook performs **Twitter Sentiment Analysis** using **TF-IDF Vectorization** and a **Multinomial Naive Bayes** classifier.

## Dataset

- Dataset: `twitter_training.csv`
- Total records: **74,681 tweets**
- Features:
  - Tweet ID
  - Entity
  - Sentiment (Target)
  - Tweet Content
- Sentiment classes:
  - Positive
  - Negative
  - Neutral
  - Irrelevant

## Data Quality

- Missing values in Tweet Content: **686**
- Valid tweet texts: **73,995**

## Workflow

1. Load Twitter sentiment dataset
2. Clean and preprocess tweet text
3. Convert text into numerical features using **TF-IDF**
4. Split data into training and testing sets
5. Train a **Multinomial Naive Bayes** model
6. Evaluate performance using:
   - Confusion Matrix
   - Accuracy Score
   - Classification Report

---

# Results

## Overall Accuracy

**66.76%**

## Classification Report

| Sentiment | Precision | Recall | F1-Score |
|------------|------------|---------|----------|
| Irrelevant | 0.78 | 0.40 | 0.53 |
| Negative | 0.63 | 0.84 | 0.72 |
| Neutral | 0.70 | 0.55 | 0.62 |
| Positive | 0.66 | 0.75 | 0.70 |

## Average Performance

- Macro F1 Score: **0.64**
- Weighted F1 Score: **0.66**

## Key Findings

- The model performs best on **Negative** tweets:
  - Recall = **84%**
  - F1 = **72%**
- **Positive** tweets are also classified reasonably well:
  - Recall = **75%**
  - F1 = **70%**
- **Irrelevant** tweets are the hardest to identify:
  - Recall = **40%**
- Overall, the model correctly classifies about **2 out of every 3 tweets**.

## Example Prediction

The notebook tests a custom tweet:

> "I dont know if I hate or love this product!"

**Predicted Sentiment:** **Negative**

---

# Conclusion

The TF-IDF + Multinomial Naive Bayes approach achieved **66.8% accuracy** on a four-class Twitter sentiment classification problem. The model is particularly effective at detecting **negative sentiment**, while performance on **neutral** and **irrelevant** tweets could be improved through better text preprocessing, hyperparameter tuning, or more advanced models such as Logistic Regression, SVM, or transformer-based NLP models.
