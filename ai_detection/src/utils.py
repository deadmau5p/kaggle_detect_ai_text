import string

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def load_train_dataset():
    train_data = pd.read_csv("./data/llm-detect-ai-generated-text/train_essays.csv")
    return train_data


def load_test_dataset():
    test_data = pd.read_csv("data/llm-detect-ai-generated-text/test_essays.csv")
    return test_data


def remove_stop_words(input_df):
    stop = stopwords.words("english")

    input_df["text"] = input_df["text"].apply(
        lambda x: " ".join([word for word in x.split() if word not in (stop)])
    )

    return input_df


def lemmatize(input_df):
    lemmatizer = WordNetLemmatizer()

    input_df["text"] = input_df["text"].apply(
        lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()])
    )

    return input_df


def flatten_list(deep_list):
    return [item for x in deep_list for item in x]


def prepare_data(data):
    no_punc_df = remove_punctuation(data)
    stop_words_removed = remove_stop_words(no_punc_df)
    lemmatized = lemmatize(stop_words_removed)

    return lemmatized


def remove_punctuation(input_df):
    regex = r"[{}]".format(string.punctuation)
    input_df["text"] = input_df["text"].str.replace(regex, "", regex=True)
    return input_df


if __name__ == "__main__":
    train_data = load_train_dataset()
    train_df = prepare_data(train_data)
    # print(train_df)
