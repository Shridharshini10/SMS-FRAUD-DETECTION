from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/model.pkl')
