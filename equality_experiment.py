from datasets import EqualityDataset, PremackDataset, PremackDatasetLeafFlattened
from itertools import product
import numpy as np
import os
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
from torch_input_as_output import InputAsOutputClassifier
from trained_datasets import TrainedEqualityDataset
from trained_datasets import TrainedPremackDatasetLeafFlattenedIntermediateSupervision
import warnings


class EqualityExperiment:

    def __init__(self,
            dataset_class=EqualityDataset,
            n_hidden=1,
            model=None,
            n_trials=10,
            train_sizes=list(range(104, 100001, 5000)),
            embed_dims=[2, 10, 25, 50, 100],
            hidden_dims=[2, 10, 25, 50, 100],
            alphas=[0.00001, 0.0001, 0.001],
            learning_rates=[0.0001, 0.001, 0.01],
            test_set_class_size=250):
        self.dataset_class = dataset_class
        self.n_hidden = n_hidden
        self.model = model
        self.n_trials = n_trials
        self.train_sizes = train_sizes
        self.class_size = int(max(self.train_sizes) / 2)
        self.embed_dims = embed_dims
        self.hidden_dims = hidden_dims
        self.alphas = alphas
        self.learning_rates = learning_rates
        grid = (self.embed_dims, self.hidden_dims, self.alphas, self.learning_rates)
        self.grid = list(product(*grid))
        self.test_set_class_size = test_set_class_size

    def run(self):
        data = []

        print(f"Grid size: {len(self.grid)} * {self.n_trials}; "
              f"{len(self.grid)*self.n_trials} experiments")

        for embed_dim, hidden_dim, alpha, lr in self.grid:

            print(f"Running trials for embed_dim={embed_dim} hidden_dim={hidden_dim} "
                  f"alpha={alpha} lr={lr} ...", end=" ")

            start = time.time()

            scores = []

            for trial in range(1, self.n_trials+1):

                mod = self.get_model(hidden_dim, alpha, lr, embed_dim)

                X_train, X_test, y_train, y_test, test_dataset = \
                  self.get_new_train_and_test_sets(embed_dim)

                # Record the result with no training if the model allows it:
                try:
                    preds = mod.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    scores.append(acc)
                    train_preds = mod.predict(X_train)
                    train_acc = accuracy_score(y_train, train_preds)
                    d = {
                        'trial': trial,
                        'train_size': 0,
                        'embed_dim': embed_dim,
                        'hidden_dim': hidden_dim,
                        'alpha': alpha,
                        'learning_rate': lr,
                        'accuracy': acc,
                        'train_accuracy': train_acc,
                        'batch_pos': 0,
                        'batch_neg': 0}
                    if hasattr(self, "pretraining_metadata"):
                        d.update(self.pretraining_metadata)
                    data.append(d)
                except NotFittedError:
                    pass

                for train_size in self.train_sizes:

                    if train_size < 40:
                        X_batch, y_batch = self.get_minimal_train_set(
                            train_size, embed_dim, test_dataset)
                        batch_pos = sum([1 for label in y_batch if label == 1])
                    else:
                        X_batch = X_train[ : train_size]
                        y_batch = y_train[ : train_size]
                        batch_pos = sum([1 for label in y_train[ : train_size] if label == 1])

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mod.fit(X_batch, y_batch)

                    # Predictions:
                    preds = mod.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    scores.append(acc)
                    train_preds = mod.predict(X_batch)
                    train_acc = accuracy_score(y_batch, train_preds)
                    d = {
                        'trial': trial,
                        'train_size': train_size,
                        'embed_dim': embed_dim,
                        'hidden_dim': hidden_dim,
                        'alpha': alpha,
                        'learning_rate': lr,
                        'accuracy': acc,
                        'train_accuracy': train_acc,
                        'batch_pos': batch_pos,
                        'batch_neg': len(X_batch) - batch_pos}
                    if hasattr(self, "pretraining_metadata"):
                        d.update(self.pretraining_metadata)
                    data.append(d)

            elapsed_time = round(time.time() - start, 0)

            print(f"mean: {round(np.mean(scores), 2)}; max: {max(scores)}; took {elapsed_time} secs")

        self.data_df = pd.DataFrame(data)
        return self.data_df

    def to_csv(self, base_output_filename, output_dirname="results"):
        self.data_df.to_csv(
            os.path.join(output_dirname, base_output_filename),
            index=None)

    def get_model(self, hidden_dim, alpha, lr, embed_dim):
        if self.model is None:
            return MLPClassifier(
                max_iter=1,
                hidden_layer_sizes=tuple([hidden_dim] * self.n_hidden),
                activation='relu',
                alpha=alpha,
                solver='adam',
                learning_rate_init=lr,
                beta_1=0.9,
                beta_2=0.999,
                warm_start=True)
        else:
            return self.model(
                hidden_dim=hidden_dim,
                alpha=alpha,
                lr=lr,
                embed_dim=embed_dim)

    def get_new_train_and_test_sets(self, embed_dim):
        train_dataset = self.dataset_class(
            embed_dim=embed_dim,
            n_pos=self.class_size,
            n_neg=self.class_size)
        X_train, y_train = train_dataset.create()

        test_dataset = self.dataset_class(
            embed_dim=embed_dim,
            n_pos=self.test_set_class_size,
            n_neg=self.test_set_class_size)
        X_test, y_test = test_dataset.create()

        train_dataset.test_disjoint(test_dataset)

        return X_train, X_test, y_train, y_test, test_dataset

    def get_minimal_train_set(self, train_size, embed_dim, other_dataset):
        class_size = int(train_size / 2)
        train_dataset = self.dataset_class(
            embed_dim=embed_dim,
            n_pos=class_size,
            n_neg=class_size)
        X_batch, y_batch = train_dataset.create()

        train_dataset.test_disjoint(other_dataset)

        return X_batch, y_batch


class PretrainedEqualityExperiment(EqualityExperiment):
    def __init__(self, n_tasks, max_iter, **kwargs):
        super().__init__(**kwargs)
        self.n_tasks = n_tasks
        self.max_iter = max_iter

    def get_new_train_and_test_sets(self, embed_dim):
        train_dataset = self.dataset_class(
            n_tasks=self.n_tasks,
            max_iter=self.max_iter,
            embed_dim=embed_dim,
            n_pos=self.class_size + self.test_set_class_size,
            n_neg=self.class_size + self.test_set_class_size)
        X, y = train_dataset.create()
        test_size = self.test_set_class_size * 2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test, train_dataset

    def get_minimal_train_set(self, train_size, embed_dim, other_dataset):
        raise RuntimeError(
            "Train sets under size 40 are not supported by "
            "`PretrainedEqualityExperiment`, since we can't "
            "ensure that we have a proper label distribution.")


class InputAsOutputExperiment(EqualityExperiment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_class = PremackDatasetLeafFlattened
        self.equality_n_pos = 40000
        self.equality_n_neg = 40000
        self.max_equality_pretraining_trials = 1

    def get_model(self, hidden_dim, alpha, lr, embed_dim):
        best_acc = 0.0
        best_mod = None
        # Keep doing equality pretraining until we have a
        # model that does perfectly on the test set or we
        # reach `self.max_equality_pretraining_trials`, at
        # which point we return the best model we found.
        for _ in range(self.max_equality_pretraining_trials):
            # Train data:
            X_train, y_train = EqualityDataset(
                n_pos=self.equality_n_pos,
                n_neg=self.equality_n_neg,
                embed_dim=embed_dim).create()
            # Model training:
            mod = InputAsOutputClassifier(
                alpha=alpha,
                eta=lr,
                warm_start=True,
                max_iter=1)
            mod.fit(X_train, y_train)
            # Test data:
            X_test, y_test = EqualityDataset(
                n_pos=self.test_set_class_size,
                n_neg=self.test_set_class_size,
                embed_dim=embed_dim).create()
            # Evaluation:
            preds = mod.predict(X_test)
            acc = accuracy_score(y_test, preds)
            # Decision making about the model:
            if acc > best_acc:
                best_mod = mod
                best_acc = acc
            if acc == 1.0:
                break
        self.pretraining_metadata = {
            "equality_pretraining_accuracy": best_acc,
            "max_equality_pretraining_trials": self.max_equality_pretraining_trials}
        return best_mod

    def to_csv(self, **kwargs):
        base_output_filename = "input-as-output.csv"
        super().to_csv(base_output_filename, **kwargs)


class PretrainedInputAsOutputExperiment(EqualityExperiment):
    def __init__(self,
            rep_pretrain_tasks,
            rep_pretrain_max_iter,
            equality_pretrain_max_iter,
            **kwargs):
        super().__init__(**kwargs)
        self.dataset_class = TrainedPremackDatasetLeafFlattenedIntermediateSupervision
        self.rep_pretrain_tasks = rep_pretrain_tasks
        self.rep_pretrain_max_iter = rep_pretrain_max_iter
        self.equality_pretrain_max_iter = equality_pretrain_max_iter
        self.max_equality_pretraining_trials = 1

    def get_model(self, hidden_dim, alpha, lr, embed_dim):
        best_acc = 0.0
        best_mod = None
        best_dataset = None

        for _ in range(self.max_equality_pretraining_trials):
            # Train data:
            dataset = self.dataset_class(
                n_tasks=self.rep_pretrain_tasks,
                max_iter=self.rep_pretrain_max_iter,
                n_pos=self.class_size + self.test_set_class_size,
                n_neg=self.class_size + self.test_set_class_size,
                embed_dim=embed_dim)
            X, y = dataset.create()

            # Train/test split:
            test_size = self.test_set_class_size * 2
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size)

            X_train_flat, y_train_flat = self.flatten_dataset(X_train, y_train)
            X_test_flat, y_test_flat = self.flatten_dataset(X_test, y_test)

            # Model training:
            mod = InputAsOutputClassifier(
                alpha=alpha,
                eta=lr,
                warm_start=True,
                max_iter=self.equality_pretrain_max_iter)
            mod.fit(X_train_flat, y_train_flat)

            # Evaluation:
            preds = mod.predict(X_test_flat)
            acc = accuracy_score(y_test_flat, preds)

            # Decision making about the model:
            if acc > best_acc:
                best_mod = mod
                best_dataset = dataset
                best_acc = acc
            # No need to continue if we did perfectly:
            if acc == 1.0:
                break
        # Set the dataset attribute for use in creating splits:
        self.dataset = dataset
        self.X_train = X_train[: max(self.train_sizes)]
        self.X_test = X_test
        self.y_train = [x[2] for x in y_train[: max(self.train_sizes)]]
        self.y_test = [x[2] for x in y_test]
        self.pretraining_metadata = {
            'equality_pretraining_accuracy': best_acc,
            'max_equality_pretraining_trials': self.max_equality_pretraining_trials,
            'equality_pretrain_max_iter': self.equality_pretrain_max_iter,
            'rep_pretrain_max_iter': self.rep_pretrain_max_iter,
            'rep_pretrain_tasks': self.rep_pretrain_tasks}
        best_mod.max_iter = 1

        return best_mod

    @staticmethod
    def flatten_dataset(X, y):
        X = np.concatenate((X[: , 0, : ], X[: , 1, : ]))
        y = [x[0] for x in y] + [x[1] for x in y]
        return X, y

    def to_csv(self, **kwargs):
        base_output_filename = "input-as-output-pretrain-{}tasks.csv".format(
            self.rep_pretrain_tasks)
        super().to_csv(base_output_filename, **kwargs)

    def get_new_train_and_test_sets(self, embed_dim):
        return self.X_train, self.X_test, self.y_train, self.y_test, self.dataset

    def get_minimal_train_set(self, train_size, embed_dim, test_dataset):
        raise RuntimeError(
            "Train sets under size 40 are not supported by "
            "`PretrainedInputAsOutputExperiment`, since we can't "
            "ensure that we have a proper label distribution.")
