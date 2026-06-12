import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import streamlit as st

def clean(text):
    text=str(text).lower()
    text=re.sub(r'@\w+', '', text)
    text=re.sub(r'\d+', '', text)
    text=re.sub(r'https?://\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    #text = re.sub(r'[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~ ]', '', text)
    return text

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv ('twitter_training.csv')

    columns = ['Tweet ID','Entity', 'Sentiment', 'Tweet Content']
    df.columns = columns

    df.fillna(df.mean(numeric_only= True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace= True)

    x = df['Tweet Content']
    y = df['Sentiment']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state= 42)

    pipe = Pipeline(
    steps=[
        #('clean', FunctionTransformer(clean)),
        ('vecro', TfidfVectorizer(stop_words= 'english', preprocessor=clean, max_features = 10000, ngram_range = (1,2))),
        ('estimator', MultinomialNB())
    ]
    )
    pipe.fit(x_train, y_train)
    return pipe

pipe = load_and_train_model()

st.title('AI Sentiment Analysis App')
st.write("Type any product review or simply comment below to find out what category is your review")

user_input = st.text_area("Enter your review here", placeholder =  "Type something like 'This product is amazing!'...")

if st.button("Enter your review"):
    if user_input.strip() == '':
        st.warning('Please wnter some text before clicking')
    else:
        processed_input =[user_input]
        prediction = pipe.predict(processed_input)
        confidence = pipe.predict_proba(processed_input)
        # Display results to the user
        st.success(f"Predicted Sentiment: {prediction[0]}")
        #st.write(f"Prediction Confidence: {max(confidence[0]) * 100:.2f}%")

