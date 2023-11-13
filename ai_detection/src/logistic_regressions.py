import pandas as pd
from bag_of_words import BagOfWords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from utils import load_test_dataset, load_train_dataset, prepare_data


def train_logistic_regression(train_data):
    X = train_data["bow_vector"].to_list()
    y = train_data["label"].to_list()

    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    return model


def make_submission(model, test_data):
    X = test_data["bow_vector"].to_list()
    print(X)
    # y = test_data["label"].to_list()

    probabilities = model.predict_proba(X)
    # print(probabilities)

    # Evaluating the Model
    # print("Accuracy:", accuracy_score(y, y_pred))
    # sprint("\nClassification Report:\n", classification_report(y, y_pred))
    print(probabilities)


if __name__ == "__main__":
    train_data = load_train_dataset()
    train_data = prepare_data(train_data)
    test_data = load_test_dataset()
    test_data = prepare_data(test_data)

    bag_of_words = BagOfWords(train_data, test_data)
    print(bag_of_words.train_data)

    model = train_logistic_regression(bag_of_words.train_data)
    print(bag_of_words.test_data)
    make_submission(model, bag_of_words.test_data)
