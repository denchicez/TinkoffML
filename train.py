import argparse
import re
import pickle
import os


class Train:
    @staticmethod
    def __read_data(input_dir: str) -> str:
        if input_dir is None:
            data = input("Read from console\n")
        else:
            data = []
            try:
                for file in os.listdir(input_dir):
                    if file.endswith(".txt"):
                        path_txt = os.path.join(input_dir, file)
                        with open(path_txt, 'r') as f:
                            data.append(f.read())
                data = ' '.join(data)
            except FileNotFoundError as e:
                print(f"Input file not found, errors is -> {e}")
        return data

    def __init__(self, args):
        self.__data = self.__read_data(args.input)
        self.__output_dir = args.model

    @staticmethod
    def __tokenize(data: str) -> list:
        data = re.sub(r'[^а-яa-zё\s]+', ' ', data.lower())
        items = data.split()
        return items

    @staticmethod
    def __create_ngram(n: int, tokens: list) -> dict:
        ngram = {}
        for i in range(len(tokens) - n):
            n_tokens = tokens[i: i + n]
            n_tokens = tuple(n_tokens)
            next_token = tokens[i + n]
            d = {next_token: ngram.get(n_tokens, {}).get(next_token, 0) + 1}
            if n_tokens not in ngram:
                ngram[n_tokens] = d
            else:
                ngram[n_tokens].update(d)

        for key in ngram.keys():
            counter_all = sum(ngram[key].values())
            for token in ngram[key]:
                ngram[key][token] = ngram[key][token] / counter_all
            ngram[key] = list(ngram[key].items())
            ngram[key].sort(key=lambda tup: tup[1], reverse=True)
        return ngram

    @staticmethod
    def __save(output_dir: str, ngrams: dict) -> None:
        try:
            with open(f'{output_dir}', 'wb') as f:
                pickle.dump(ngrams, f)
        except FileNotFoundError as e:
            print(f'Output path to save model not found. Error is {e}')

    def fit(self, n: int = 1) -> None:
        if n < 1:
            raise "N must be more or equals then 1"
        n -= 1
        tokenize_data = self.__tokenize(self.__data)
        ngrams = self.__create_ngram(n, tokenize_data)
        self.__save(self.__output_dir, ngrams)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for model train')
    parser.add_argument('--input-dir', dest="input", type=str, help='path to collection data')
    parser.add_argument('--model', dest="model", type=str, help='path to save model', required=True)
    args = parser.parse_args()

    train = Train(args)
    train.fit(n=2)  # bi-gramm
