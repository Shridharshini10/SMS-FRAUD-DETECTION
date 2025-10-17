from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def preprocess_data(path):
    df = pd.read_csv(path, sep='\t', names=['label', 'message'])
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['message'])
    y = df['label'].map({'ham': 0, 'spam': 1})
    return X, y, vectorizer
