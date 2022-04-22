from utils import randvec
import numpy as np
import random
from torch_rep_learner import RepLearner


class TrainedEqualityDataset:

    POS_LABEL = 1
    NEG_LABEL = 0

    def __init__(self,
            embed_dim=50,
            n_pos=500,
            n_neg=500,
            flatten=True,
            n_tasks=3,
            max_iter=10,
            hidden_dim=None):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or self.embed_dim
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.flatten = flatten
        self.n_tasks = n_tasks
        self.max_iter = max_iter
        # We need twice as many negative examples to achieve label balance
        # because, for them, we pair different vectors:
        self.vocab_size = self.n_pos + (self.n_neg * 2)

    def create(self):
        self._train_embedding()
        pos_embed = self.embedding[: self.n_pos]
        neg_embed = self.embedding[self.n_pos: ]
        self.data = []
        self.data += self._create_pos(pos_embed)
        self.data += self._create_neg(neg_embed)
        random.shuffle(self.data)
        data = self.data.copy()
        if self.flatten:
            data = [(np.concatenate(x), label) for x, label in data]
        X, y = zip(*data)
        self.X = np.array(X)
        self.y = y
        return self.X, self.y

    def _train_embedding(self):
        embedding = np.array([randvec(self.embed_dim) for _ in range(self.vocab_size)])
        mod = RepLearner(
            self.vocab_size,
            embed_dim=self.embed_dim,
            embedding=embedding,
            hidden_dim=self.hidden_dim,
            n_tasks=self.n_tasks,
            max_iter=self.max_iter)
        X = list(range(self.vocab_size))
        ys = []
        for _ in range(self.n_tasks):
            y = np.random.choice((0, 1), size=self.vocab_size, replace=True)
            ys.append(y)
        ys = list(zip(*ys))
        mod.fit(X, ys)
        self.embedding = mod.embedding
        self.embedding_labels = ys

    def _create_pos(self, examples):
        data = []
        for vec in examples:
            ex = ((vec, vec), self.POS_LABEL)
            data.append(ex)
        return data

    def _create_neg(self, examples):
        data = []
        for i in range(0, len(examples)-1, 2):
            ex = ((examples[i], examples[i+1]), self.NEG_LABEL)
            data.append(ex)
        return data

    def test_disjoint(self, other_dataset):
        these_vecs = {tuple(x) for pair, label in self.data for x in pair}
        other_vecs = {tuple(x) for pair, label in other_dataset.data for x in pair}
        shared = these_vecs & other_vecs
        assert len(shared) == 0, \
            f"This dataset and the other dataset shared {len(shared)} word-level reps."


class TrainedPremackDataset(TrainedEqualityDataset):
    def __init__(self,
            embed_dim=50,
            n_pos=500,
            n_neg=500,
            n_tasks=3,
            max_iter=10,
            hidden_dim=None,
            flatten_root=True,
            flatten_leaves=True,
            intermediate=False):
        super().__init__(
            embed_dim=embed_dim,
            n_pos=n_pos,
            n_neg=n_neg,
            n_tasks=n_tasks,
            max_iter=max_iter,
            hidden_dim=hidden_dim)

        for n, v in ((n_pos, 'n_pos'), (n_neg, 'n_neg')):
            if n % 2 != 0:
                raise ValueError(
                    f"The value of {v} must be even to ensure a balanced "
                    f"split across its two sub-types of the {v} class.")

        self.n_same_same = int(n_pos / 2)
        self.n_diff_diff = int(n_pos / 2)
        self.n_same_diff = int(n_neg / 2)
        self.n_diff_same = int(n_neg / 2)
        self.flatten_root = flatten_root
        self.flatten_leaves = flatten_leaves
        self.intermediate = intermediate

        # The multiplications correspond to how many distinct vectors
        # we need for each example type in the Premack setting. For
        # example, same/diff uses 1 vector for the same part and 2
        # for the diff part.
        self.vocab_size = (self.n_pos * 2) + (self.n_neg * (4 + 3 + 3))
        self.holdout = tuple([1] * self.n_tasks)

    def create(self):
        self._train_embedding()

        # The multiplications correspond to how many distinct vectors
        # we need for example type:
        b1 = self.n_same_same * 2
        b2 = b1 + (self.n_diff_diff * 4)
        b3 = b2 + (self.n_same_diff * 3)
        b4 = b3 + (self.n_diff_same * 3)

        self.data = []

        if self.intermediate:
            self.data += [(x, [self.POS_LABEL, self.POS_LABEL, y]) for x, y in self._create_same_same(self.embedding[ : b1])]
            self.data += [(x, [self.NEG_LABEL, self.NEG_LABEL, y]) for x, y in self._create_diff_diff(self.embedding[b1: b2])]
            self.data += [(x, [self.POS_LABEL, self.NEG_LABEL, y]) for x, y in self._create_same_diff(self.embedding[b2: b3])]
            self.data += [(x, [self.NEG_LABEL, self.POS_LABEL, y]) for x, y in self._create_diff_same(self.embedding[b3: b4])]
        else:
            self.data += self._create_same_same(self.embedding[ : b1])
            self.data += self._create_diff_diff(self.embedding[b1: b2])
            self.data += self._create_same_diff(self.embedding[b2: b3])
            self.data += self._create_diff_same(self.embedding[b3: b4])

        random.shuffle(self.data)

        data = self.data.copy()

        if self.flatten_root or self.flatten_leaves:
            data = [((np.concatenate(x1), np.concatenate(x2)), label)
                    for (x1, x2), label in data]
        if self.flatten_root:
            data = [(np.concatenate(x), label) for x, label in data]

        X, y = zip(*data)
        self.X = np.array(X)
        self.y = y

        return self.X, self.y

    def test_disjoint(self, other_dataset):
        these_vecs = {tuple(x) for root_pair, label in self.data
                               for pair in root_pair for x in pair}
        other_vecs = {tuple(x) for root_pair, label in other_dataset.data
                               for pair in root_pair for x in pair}
        shared = these_vecs & other_vecs
        assert len(shared) == 0, \
            f"This dataset and the other dataset shared {len(shared)} word-level reps."

    def _create_same_same(self, examples):
        data = []
        for i in range(0, len(examples)-1, 2):
            left = (examples[i], examples[i])
            right = (examples[i+1], examples[i+1])
            rep = (left, right)
            data.append((rep, self.POS_LABEL))
        return data

    def _create_diff_diff(self, examples):
        data = []
        for i in range(0, len(examples)-3, 4):
            left = (examples[i], examples[i+1])
            right = (examples[i+2], examples[i+3])
            rep = (left, right)
            data.append((rep, self.POS_LABEL))
        return data

    def _create_same_diff(self, examples):
        data = []
        for i in range(0, len(examples)-2, 3):
            left = (examples[i], examples[i])
            right = (examples[i+1], examples[i+2])
            rep = (left, right)
            data.append((rep, self.NEG_LABEL))
        return data

    def _create_diff_same(self, examples):
        data = []
        for i in range(0, len(examples)-2, 3):
            left = (examples[i+1], examples[i+2])
            right = (examples[i], examples[i])
            rep = (left, right)
            data.append((rep, self.NEG_LABEL))
        return data


class TrainedPremackDatasetLeafFlattenedIntermediateSupervision(TrainedPremackDataset):
    def __init__(self, embed_dim=50, hidden_dim=50,
            n_pos=500, n_neg=500, n_tasks=3, max_iter=10):

        super().__init__(
            embed_dim=embed_dim,
            n_pos=n_pos,
            n_neg=n_neg,
            n_tasks=n_tasks,
            max_iter=max_iter,
            hidden_dim=hidden_dim,
            flatten_root=False,
            flatten_leaves=True,
            intermediate=True)
