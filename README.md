# Relational reasoning and generalization using non-symbolic neural networks

This is the code repository for

> Geiger, Atticus; Alexandra Carstensen; Michael C. Frank; and Christopher Potts. 2020. 'Relational reasoning and generalization using non-symbolic neural networks'.  Ms., Stanford University.


## Requirements

This code requires Python 3.6 or higher. Specific requirements are given in `requirements.txt`. For installing [PyTorch](https://pytorch.org), we recommend following the specific instructions provided at this project's website.

## Pretraining

The pretraining model is `torch_rep_learner.py`.


## Datasets

* `datasets.py`: Classes for creating equality datasets that do not involve pretraining.

* `trained_datasets.py`: Counterparts of the classes in `datasets.py` but with pretrained embedding matrices.

* (The dataset code for Model 2 is included in `fuzzy_lm_experiment.py`.)



## Model 1: Same-different relations

* The core model is an `sklearn.neural_network.MLPClassifier`.

* `equality_experiment.py` is a framework for running the experiments, including all hyperparameter tuning.

* The experiment runs are in `run_basic_equality.ipynb`.

* The results are `results/equality.csv`, `results/equality-pretrain-3tasks.csv`, `results/equality-pretrain-5tasks.csv`, and `results/equality-pretrain-10tasks.csv`.


## Model 2: Sequential same-different (ABA task)

* The core model is a language model with a mean-squared error loss, implemented in PyTorch. The code is `torch_fuzzy_lm.py`.

* `relu_lstm.py` is a PyTorch LSTM with a ReLU activation. This is used by `torch_fuzzy_lm.py`.

* `fuzzy_lm_experiment.py` is a framework for running the experiments, including all hyperparameter tuning.

* The experiment runs are in `run_fuzzy_lm.ipynb`.

* The results are `results/fuzzy-lm-vocab20.csv`, `fuzzy-lm-vocab20-pretrain-3tasks.csv`, `fuzzy-lm-vocab20-pretrain-5tasks.csv`, and `fuzzy-lm-vocab20-pretrain-10tasks.csv`.


## Model 3: Hierarchical same-different relations

* For the train-from-scratch versions:
  * The core model is an `sklearn.neural_network.MLPClassifier`, as in Model 1.
  * `run_flat_premack.ipynb` runs the experiments, using `equality_experiment.py`.
  * The results are `results/flatpremack-h1.csv` (one hidden later) and `results/flatpremack-h2.csv` (two hidden layers).
  * The versions with pretraining are `flatpremack-h2-pretrain-3tasks.csv`, `flatpremack-h2-pretrain-5tasks.csv`, and `flatpremack-h2-pretrain-10tasks.csv`.

* For the pretraining regime:
  * The core model is `torch_input_as_output.py`.
  * `run_input_as_output.ipynb` runs the experiments, including all hyperparameter tuning.
  * The results are `results/input-as-output.csv`, along with exploratory runs involving pretraining the inputs as well: `input-as-output-pretrain-3tasks.csv`, `input-as-output-pretrain-5tasks.csv`, and `input-as-output-pretrain-10tasks.csv`.


## Visualization

The notebook `create_visualizations.ipynb` runs all the visualization, using `comparative_viz.py` and the results files in `results`. The visualizations are written to the `fig` directory.


## Other files

* `utils.py`: general shared functionality.

* `view_best_hyperparameters.ipynb`: can be used to see which hyperparameters are optimal for the experiments, using the files in `results`.

* `rmts.mplstyle`: matplotlib style file for visualizations. This needs to be placed in the directory that matplotlib looks for such files, which depends somewhat on the system: https://matplotlib.org/users/style_sheets.html

* The `test` directory contains unit tests.
