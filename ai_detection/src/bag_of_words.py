import numpy as np
import pandas as pd
from utils import flatten_list, load_test_dataset, load_train_dataset, prepare_data


class BagOfWords:
    def __init__(self, train_data, test_data) -> None:
        self.test_data = test_data
        self.test_data["generated"] = 0
        self.train_data = train_data
        self.dataframe = pd.concat([train_data, test_data], axis=0, ignore_index=True)
        self.vocab = self.build_vocab()
        self.encode_dataset()

    def make_bow_vector(self, split_text: list):
        bow_vector = np.zeros(len(self.vocab.keys()), dtype=int)
        for word in split_text:
            bow_vector[self.vocab[word]] += 1

        return bow_vector

    def encode_dataset(self):
        self.dataframe["bow_vector"] = self.dataframe["split_text"].apply(
            lambda split_text: self.make_bow_vector(split_text)
        )

        self.dataframe = self.dataframe.drop(
            ["id", "prompt_id", "text", "split_text"], axis=1
        )
        self.dataframe = self.dataframe.rename({"generated": "label"}, axis=1)

        return self.dataframe

    def build_vocab(self):
        self.dataframe["split_text"] = self.dataframe["text"].apply(
            lambda text: text.split(" ")
        )
        list_of_texts = self.dataframe["split_text"].to_list()
        unique_words = set(flatten_list(list_of_texts))

        vocab = {word: i for i, word in enumerate(unique_words)}

        return vocab


if __name__ == "__main__":
    train_data = load_train_dataset()
    test_data = load_test_dataset()
    train_df = prepare_data(train_data)
    test_df = prepare_data(test_data)
    bag_of_words = BagOfWords(train_df, test_data)
    print(bag_of_words.dataframe)
