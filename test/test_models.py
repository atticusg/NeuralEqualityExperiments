from datasets import EqualityDataset, PremackDataset
from fuzzy_lm_experiment import DatasetABA
import numpy as np
import pytest
import string
from torch_fuzzy_lm import FuzzyPatternLM, START_SYMBOL, END_SYMBOL
from torch_input_as_output import InputAsOutputClassifier
from torch_rep_learner import RepLearner


def test_fuzzy_pattern_lm():
    train_vocab = list(string.ascii_lowercase)
    test_vocab = list(string.ascii_lowercase)
    vocab = train_vocab + test_vocab + [START_SYMBOL, END_SYMBOL]
    dataset = DatasetABA(train_vocab, test_vocab)
    mod = FuzzyPatternLM(vocab, embed_dim=2, hidden_dim=3, max_iter=1)
    mod.fit(dataset.train)
    mod.predict_one(dataset.test[0])


def test_rep_learner():
    mod = RepLearner(vocab_size=4, embed_dim=2, hidden_dim=3)
    X = list(range(4))
    ys = [(1,1), (1,0), (0,1), (0,0)]
    mod.fit(X, ys)
    mod.predict(X)


def test_input_as_output_classifier():
    mod = InputAsOutputClassifier()
    # Basic equality pretraining:
    X_eq, y_eq = EqualityDataset(n_pos=40, n_neg=40, flatten=True).create()
    mod.fit(X_eq, y_eq)
    mod.predict(X_eq)
    # Now the hierarchical version:
    X_premack, y_premack = PremackDataset(
        n_pos=80, n_neg=80, flatten_leaves=True, flatten_root=False).create()
    mod.fit(X_premack, y_premack)
    mod.predict(X_premack)
