import argparse
import pickle

import numpy as np
import re
from time import time


class Model:
    def __init__(self):
        self.model = None

    def fit(self, model_path: str) -> None:
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found, this error -> {e}")

    @staticmethod
    def __tokenize(data: str) -> list:
        data = re.sub(r'[^а-яa-zё\s]+', ' ', data.lower())
        items = data.split()
        return items

    def __create_prefix(self, prefix: str) -> tuple:  # :NOTE: not working for all examples
        if prefix is None:  # prefix not found
            list_of_prefix = list(self.model.keys())
            prefix_index = np.random.randint(len(list_of_prefix))
            prefix = list_of_prefix[prefix_index]
        else:
            prefix = self.__tokenize(prefix)
            len_prefix = len(list(self.model.keys())[0])
            prefix = prefix[len(prefix) - len_prefix: len(prefix)]
            prefix = tuple(prefix)
        return prefix

    def __get_next_token(self, prefix: tuple) -> (str, bool):
        flag = False
        if prefix not in self.model:
            list_of_prefix = list(self.model.keys())
            prefix_index = np.random.randint(len(list_of_prefix))
            prefix = list_of_prefix[prefix_index]
            flag = True
        tokens, percent = zip(*self.model[prefix])

        token = np.random.choice(tokens, p=percent)
        return token, flag

    @staticmethod
    def __next_prefix(prefix: tuple, token: str) -> tuple:
        prefix = list(prefix)
        prefix.pop(0)
        prefix.append(token)
        return tuple(prefix)

    def generate(self, prefix: str, length: int) -> str:
        if self.model is None:
            raise Exception("model not fit")

        prefix = self.__create_prefix(prefix)
        new_sentence = []
        while len(new_sentence) < length:
            token, flag = self.__get_next_token(prefix)
            if flag:
                if len(new_sentence) > 0:
                    new_sentence[-1] += '.'
            prefix = self.__next_prefix(prefix, token)
            new_sentence.append(token)
        return ' '.join(new_sentence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for model train')
    parser.add_argument('--model', dest="model", type=str, help='path to save model', required=True)
    parser.add_argument('--prefix', dest="prefix", type=str, help='prefix of data')
    parser.add_argument('--length', dest="length", type=int, help='length of out', required=True)

    args = parser.parse_args()
    model = Model()
    model.fit(args.model)

    seed = np.random.randint(time())
    np.random.seed(seed)
    print("SEED: ", seed)

    answer = model.generate(args.prefix, args.length)
    if args.prefix is not None:
        print(args.prefix, end=" ")
    print(answer)
