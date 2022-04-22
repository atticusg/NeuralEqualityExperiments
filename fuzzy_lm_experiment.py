from itertools import product
import numpy as np
import os
import pandas as pd
import random
from relu_lstm import ReLULSTM
import string
import time
from torch_rep_learner import RepLearner
import torch.nn as nn
from torch_fuzzy_lm import FuzzyPatternLM, START_SYMBOL, END_SYMBOL
import utils


class Dataset:
    def __init__(self, train_vocab, test_vocab):
        self.train_vocab = train_vocab
        self.test_vocab = test_vocab
        self.train = self.generate(self.train_vocab)
        self.test = self.generate(self.test_vocab)

    def generate(self, vocab):
        dataset = []
        for ex in self.example_generator(vocab):
            dataset.append([START_SYMBOL] + ex + [END_SYMBOL])
        return dataset


class DatasetABA(Dataset):

    @staticmethod
    def example_generator(vocab):
        for c1 in vocab:
            for c2 in vocab:
                if c1 != c2:
                    yield [c2, c1, c2]

    @staticmethod
    def is_error(p, test_len):
        return len(p) != test_len or p[1] != p[-2] or p[1] == p[2]


class FuzzyPatternLMExperiment:
    def __init__(self,
            dataset_class=DatasetABA,
            n_trials=20,
            embed_dims=[2, 10, 25, 50, 100],
            hidden_dims=[2, 10, 25, 50, 100],
            learning_rates=[0.0001, 0.001, 0.01],
            alphas=[0.00001, 0.0001, 0.001],
            rnn_cell_class=ReLULSTM,
            num_layers=1,
            pretrain_tasks=None,
            pretrain_max_iter=10,
            dropout=0,
            max_iter=150,
            train_vocab_size=20,
            test_vocab=list(string.ascii_letters)):
        self.dataset_class = dataset_class
        self.n_trials = n_trials
        self.embed_dims = embed_dims
        self.hidden_dims = hidden_dims
        self.rnn_cell_class = rnn_cell_class
        self.pretrain_tasks = pretrain_tasks
        self.pretrain_max_iter = pretrain_max_iter
        self.max_iter = max_iter
        self.learning_rates = learning_rates
        self.alphas = alphas
        self.num_layers = num_layers
        self.dropout = dropout
        self.train_vocab_size = train_vocab_size
        self.train_vocab = list(map(str, range(self.train_vocab_size)))
        self.test_vocab = test_vocab
        self.dataset = self.dataset_class(self.train_vocab, self.test_vocab)
        self.full_vocab = self.train_vocab + self.test_vocab
        self.full_vocab += [START_SYMBOL, END_SYMBOL]

        grid = (self.embed_dims, self.hidden_dims, self.alphas, self.learning_rates)
        self.grid = list(product(*grid))

    def run(self):
        data = []

        print(f"Grid size: {len(self.grid)} * {self.n_trials}; "
              f"{len(self.grid)*self.n_trials} experiments")

        for embed_dim, hidden_dim, alpha, lr in self.grid:

            scores = []

            print(f"Running trials for embed_dim={embed_dim} hidden_dim={hidden_dim} "
                  f"alpha={alpha} learning_rate={lr} ...", end=" ")

            start = time.time()

            for trial in range(1, self.n_trials+1):

                if self.pretrain_tasks is not None:
                    embedding = self._create_pretrained_embedding(embed_dim)
                else:
                    embedding = None

                model = FuzzyPatternLM(
                    vocab=self.full_vocab,
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    rnn_cell_class=self.rnn_cell_class,
                    max_iter=self.max_iter,
                    num_layers=self.num_layers,
                    embedding=embedding,
                    dropout=self.dropout,
                    warm_start=True,
                    alpha=alpha,
                    eta=lr)

                model.fit(self.dataset.train, eval_func=self.evaluate)

                preds = model.results.copy()

                scores.append(preds[-1]['accuracy'])

                model.results = []
                model.fit(self.dataset.train, eval_func=self.evaluate_train)
                train_preds = model.results.copy()

                for p, tp in zip(preds, train_preds):
                    p['train_size'] = p['iteration'] * len(self.dataset.train)
                    p.update({
                        'trial': trial,
                        'embed_dim': embed_dim,
                        'hidden_dim': hidden_dim,
                        'train_vocab_size': self.train_vocab_size,
                        'alpha': alpha,
                        'learning_rate': lr,
                        'max_iter': self.max_iter,
                        'train_accuracy': tp['train_accuracy']})

                data += preds

            elapsed_time = round(time.time() - start, 0)

            print(f"mean: {round(np.mean(scores), 2)}; max: {max(scores)}; took {elapsed_time} secs")

        df = pd.DataFrame(data)

        df.drop(['correct', 'incorrect', 'n_correct', 'n_incorrect'], axis=1, inplace=True)

        self.data_df = df

        return self.data_df

    def to_csv(self, output_dirname="results"):
        base_output_filename = "fuzzy-lm-vocab{}".format(self.train_vocab_size)
        if 'LSTM' in self.rnn_cell_class.__class__.__name__:
            base_output_filename += "-lstm"
        if self.pretrain_tasks is not None:
            base_output_filename += "-pretrain-{}tasks".format(self.pretrain_tasks)
        base_output_filename += ".csv"
        self.data_df.to_csv(
            os.path.join(output_dirname, base_output_filename),
            index=None)

    def _create_pretrained_embedding(self, embed_dim):
        vocab_size = len(self.full_vocab)
        mod = RepLearner(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=embed_dim,
            n_tasks=self.pretrain_tasks,
            max_iter=self.pretrain_max_iter)
        X = list(range(vocab_size))
        ys = []
        for _ in range(self.pretrain_tasks):
            y = np.random.choice((0, 1), size=vocab_size, replace=True)
            ys.append(y)
        ys = list(zip(*ys))
        mod.fit(X, ys)
        return mod.embedding

    def evaluate(self, model, verbose=False):
        return self._evaluate(model, self.dataset.test, split_key="", verbose=verbose)

    def evaluate_train(self, model, verbose=False):
        return self._evaluate(model, self.dataset.train, split_key="train_", verbose=verbose)

    def _evaluate(self, model, split, split_key, verbose=False):
        test_len = len(split[0])
        prompts = sorted({tuple(ex[: 2]) for ex in split})
        all_preds = set()
        for prompt in prompts:
            pred = tuple(model.predict_one(prompt))
            all_preds.add(pred)
        corr_key = "{}correct".format(split_key)
        n_corr_key = '{}n_correct'.format(split_key)
        incorr_key = "{}incorrect".format(split_key)
        n_incorrect_key = '{}n_incorrect'.format(split_key)
        accuracy_key = "{}accuracy".format(split_key)
        data = {
            corr_key: [],
            incorr_key: []}
        for p in all_preds:
            if self.dataset.is_error(p, test_len):
                data[incorr_key].append(p)
            else:
                data[corr_key].append(p)
        data[n_corr_key] = len(data[corr_key])
        data[n_incorrect_key] = len(data[incorr_key])
        data[accuracy_key] = data[n_corr_key] / len(all_preds)
        if verbose:
            print(f"{data[n_incorrect_key]} errors for {len(all_preds)} "
                  f"test examples; accuracy is {data[accuracy_key]}")
        return data
