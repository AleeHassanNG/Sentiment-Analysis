# Twitter Sentiment Analysis using TF-IDF and Naive Bayes

## Project Overview
This project performs sentiment analysis on Twitter/X text data using Natural Language Processing (NLP) techniques and a Multinomial Naive Bayes classifier.

The notebook:
- Loads a Twitter sentiment dataset (`twitter_training.csv`)
- Performs data cleaning and preprocessing
- Handles missing values
- Converts text into numerical features using TF-IDF Vectorization
- Trains a Multinomial Naive Bayes model
- Predicts sentiment labels for new tweets
- Demonstrates a Pipeline-based implementation for easier deployment

## Dataset
The dataset contains:
- Tweet ID
- Entity
- Sentiment
- Tweet Content

Target variable:
- Sentiment

Typical sentiment classes may include:
- Positive
- Negative
- Neutral
- Irrelevant

## Data Preprocessing
The notebook applies several text-cleaning steps:
- Convert text to lowercase
- Remove usernames (@mentions)
- Remove URLs
- Remove numbers
- Remove punctuation
- Handle missing values

## Machine Learning Workflow

### Feature Engineering
Text is transformed using:
- TF-IDF (Term Frequency–Inverse Document Frequency)

### Model
The classifier used is:
- Multinomial Naive Bayes

### Pipeline Version
A Scikit-learn Pipeline is included to combine:
1. Text preprocessing
2. TF-IDF vectorization
3. Model training

This simplifies model deployment and prediction on new data.

## Example Prediction
The notebook demonstrates sentiment prediction on custom text such as:

> "I dont know if I hate or love this product!"

The trained model predicts the most likely sentiment class.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Regular Expressions (re)
- NLP Text Processing

## How to Run

1. Install required packages:
```bash
pip install pandas numpy scikit-learn
```

2. Place `twitter_training.csv` in the project directory.

3. Open and run:
```bash
jupyter notebook sentiment.ipynb
```

## Future Improvements
- Hyperparameter tuning
- Cross-validation
- Model comparison (Logistic Regression, SVM, Random Forest)
- Streamlit deployment
- Model performance dashboard using Power BI

## Author
Sentiment Analysis Project developed for NLP and Machine Learning practice using TF-IDF and Naive Bayes.
