import os
import pandas as pd
import pytest

from comparative_viz import ComparativeViz


MAX_COLS = ['alpha', 'learning_rate']


@pytest.fixture
def df():
    return pd.read_csv(
        os.path.join("results", "equality.csv"), index_col=None)


@pytest.mark.parametrize("fixed_col_val, embed_dim, train_size", [
    [50, 10, 104],
    [10, 50, 104],
])
def test_comparative_viz_value_calculation(fixed_col_val, embed_dim, train_size, df):

    viz = ComparativeViz(
        df,
        experiment_type="equality",
        secondary_col="embed_dim",
        max_cols=MAX_COLS,
        output_dirname="tmp",
        fixed_col_val=fixed_col_val,
        errorbars=False)
    mean_accuracies = viz.create(to_file=False)

    group_df = df[(df.embed_dim == embed_dim) & (df.hidden_dim == fixed_col_val)]

    alpha, lr = group_df.groupby(MAX_COLS).apply(lambda x: x.accuracy.mean()).idxmax()

    opt_df = group_df[(group_df.alpha == alpha) & (group_df.learning_rate == lr)]

    ex_df = opt_df[opt_df.train_size == train_size]

    mu = ex_df['accuracy'].mean()

    assert mu == mean_accuracies.loc[embed_dim, train_size]
