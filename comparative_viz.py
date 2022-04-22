from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import scikits.bootstrap as bootstrap
import utils
import warnings


plt.style.use("rmts.mplstyle")


class ComparativeViz:

    def __init__(self,
            df_or_base_filename,
            experiment_type="equality",
            secondary_col="embed_dim",
            accuracy_col="accuracy",
            train_size_col="train_size",
            title="",
            fixed_col_val=None,
            max_cols=['alpha', 'learning_rate'],
            max_cols_method='mean',
            errorbars=True,
            xlim=None,
            ylim=[0.46, 1.01],
            train_size_max=None,
            output_dirname="fig",
            xlabel="Train examples",
            ylabel="Mean accuracy (20 runs)",
            legend_placement="upper left",
            xtick_interval=None,
            src_dirname="results",
            colors=None):
        self.src_dirname = src_dirname
        if isinstance(df_or_base_filename, str):
            self.experiment_type = df_or_base_filename.replace(".csv", "")
            src_filename = os.path.join(self.src_dirname, df_or_base_filename)
            self.df = pd.read_csv(src_filename)
        else:
            self.df = df_or_base_filename
            self.experiment_type = experiment_type
        self.secondary_col = secondary_col
        self._fixed_col_val = fixed_col_val
        self.train_size_col = train_size_col
        self._title = title
        self.accuracy_col = accuracy_col
        self.max_cols = max_cols
        self.max_cols_method = max_cols_method
        self.errorbars = errorbars
        self._set_texts()
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xtick_interval = xtick_interval
        if self.xtick_interval is not None:
            self._set_xticks()
        self.xlim = xlim
        self.ylim = ylim
        self.train_size_max = train_size_max
        self.legend_placement = legend_placement
        self.output_dirname = output_dirname
        if colors is None:
            self.COLORS = utils.STYLE_COLORS
        else:
            self.COLORS = colors

    @property
    def fixed_col_val(self):
        return self._fixed_col_val

    @fixed_col_val.setter
    def fixed_col_val(self, val):
        self._fixed_col_val = val
        self._set_texts()

    def create(self, to_file=True):
        fig, ax = plt.subplots(figsize=(9, 6))
        colorcycle = cycle(self.COLORS)

        if self.fixed_col_val is not None:
            df = self.df[self.df[self.fixed_col] == self.fixed_col_val]
        else:
            df = self.df

        if self.train_size_max is not None:
            df = df[df[self.train_size_col] <= self.train_size_max]

        mean_accuracies = df.groupby(self.secondary_col, sort=False).apply(
            lambda group_df: self._plot_secondary(
                group_df, ax, color=next(colorcycle)))

        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_ylim(self.ylim)
        if self.xtick_interval is not None:
            ax.set_xticks(self.xticks)
        if to_file:
            self._to_file()

        return mean_accuracies

    def create_all(self):
        self.fixed_col_val = None
        fixeds = sorted(self.df[self.fixed_col].unique())
        for val in fixeds:
            self.fixed_col_val = val
            self.create()

    def _plot_secondary(self, group_df, ax, color):
        name = group_df.name

        if self.max_cols is not None:
            if self.max_cols_method == 'smallest':
                group_df = self._get_best_values_from_smallest_train_size_col(group_df)
            else:
                group_df = self._get_best_vals(group_df)
        grp = group_df.groupby(self.train_size_col)
        grp_acc = grp[self.accuracy_col]
        mu = grp_acc.mean()
        ax.plot(mu.index, mu, color=color, lw=2, label=name)
        if self.errorbars:
            upper, lower = self._bootstrap_errbars(grp_acc)
            ax.fill_between(mu.index, lower, upper, color=color, alpha=0.2)
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        ax.legend(loc=self.legend_placement)
        return mu

    def _get_best_vals(self, group_df):
        maxes = group_df.groupby(self.max_cols).apply(
            lambda x: x[self.accuracy_col].mean()).idxmax()
        for colname, val in zip(self.max_cols, maxes):
            group_df = group_df[group_df[colname] == val]
        return group_df

    def _get_best_values_from_smallest_train_size_col(self, group_df):
        min_train_size = group_df[self.train_size_col].min()
        zero = group_df[group_df[self.train_size_col] == min_train_size]
        maxes = zero.groupby(self.max_cols).apply(
            lambda x: x[self.accuracy_col].mean()).idxmax()
        for colname, val in zip(self.max_cols, maxes):
            group_df = group_df[group_df[colname] == val]
        return group_df

    def _to_file(self):
        output_filename = (
            f"{self.experiment_type}-{self.train_size_col}-"
            f"{self.secondary_col}-{self.fixed_col}={self.fixed_col_val}.pdf")
        if 'train' in self.accuracy_col:
            output_filename = "train-" + output_filename
        output_filename = os.path.join(self.output_dirname, output_filename)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=200)

    def _set_texts(self):
        if self.secondary_col == "embed_dim":
            self.fixed_col = "hidden_dim"
        else:
            self.fixed_col = "embed_dim"
        if self.secondary_col == "embed_dim":
            self.title = "Embedding dimensionality"
            self.fixed_label = "hidden"
        else:
            self.title = "Hidden dimensionality"
            self.fixed_label = "embedding"
        if self.fixed_col_val is not None:
            self.title += f"; {self.fixed_label} = {self.fixed_col_val}"
        if self._title is not None:
            self.title = self._title

    def _set_xticks(self):
        xtick_vals = self.df[self.train_size_col]
        self.xticks = list(np.arange(xtick_vals.min(), xtick_vals.max()+1, self.xtick_interval))
        if xtick_vals.max() not in self.xticks:
           self.xticks.append(xtick_vals.max())

    @staticmethod
    def _bootstrap_errbars(accuracy_df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            upper, lower = zip(*accuracy_df.apply(bootstrap.ci))
        return lower, upper



def compare_with_and_without_pretraining_viz(
        nopretrain_base_filename,
        pretrain_base_filenames,
        embed_dim,
        hidden_dim,
        nopretrain_color,
        experiment_type,
        accuracy_col="accuracy",
        train_size_max=None,
        dirname="results",
        ylim=[0.46, 1.01],
        xlabel="Train examples",
        max_cols_method="mean",
        legend_placement="lower right"):

    COLORS = [nopretrain_color]
    COLORS += utils.ALT_COLORS[: len(pretrain_base_filenames)]

    nopretrain_filename = os.path.join(dirname, nopretrain_base_filename)
    pretrain_filenames = [os.path.join(dirname, n) for n in pretrain_base_filenames]

    dfs = []

    def filter_dataframe(x):
        x = x[x.embed_dim == embed_dim]
        if hidden_dim is not None:
            x = x[x.hidden_dim == hidden_dim]
        return x

    nopre_df = pd.read_csv(nopretrain_filename, index_col=None)
    nopre_df = filter_dataframe(nopre_df)
    nopre_df['pretrained'] = "no pretraining"

    dfs.append(nopre_df)

    for filename in pretrain_filenames:
        n_tasks = re.search(r"(\d+)tasks", filename).group(1)
        pre_df = pd.read_csv(filename, index_col=None)
        pre_df = filter_dataframe(pre_df)
        pre_df['pretrained'] = "{}-task pretraining".format(n_tasks)
        dfs.append(pre_df)

    df = pd.concat(dfs, sort=True)

    viz = ComparativeViz(
        df,
        experiment_type=experiment_type,
        secondary_col="pretrained",
        accuracy_col=accuracy_col,
        train_size_col="train_size",
        max_cols_method=max_cols_method,
        title="",
        xlabel=xlabel,
        xtick_interval=None,
        ylim=ylim,
        max_cols=['alpha', 'learning_rate'],
        legend_placement=legend_placement,
        train_size_max=train_size_max,
        colors=COLORS)

    viz.create()


def input_as_output_zero_shot_viz(base_filename, dirname="results", output_dirname="fig"):
    src_filename = os.path.join(dirname, base_filename)

    df = pd.read_csv(src_filename, index_col=None)

    output_filename = base_filename.replace(".csv", "-zero-shot.pdf")
    output_filename = os.path.join(output_dirname, output_filename)

    df = df[df.train_size == 0.0]

    def per_group_optimal(group_df):
        alpha, lr = group_df.groupby(
            ['alpha', 'learning_rate']).apply(
                lambda x: x['accuracy'].mean()).idxmax()
        group_df = group_df[group_df.alpha == alpha]
        group_df = group_df[group_df.learning_rate == lr]
        return group_df

    df = df.groupby(['embed_dim']).apply(per_group_optimal).reset_index(drop=True)

    acc = df.groupby('embed_dim')['accuracy']

    lower, upper = ComparativeViz._bootstrap_errbars(acc)

    mu = acc.mean()

    ax = mu.plot.bar(color=utils.STYLE_COLORS, yerr=[mu-lower, upper-mu])

    for i, ((x_pos, val), u) in enumerate(zip(mu.items(), lower)):
        ax.annotate("{0:.02}".format(val), (i, u+0.1), va="top", ha="center", fontsize=12)

    ax.set_xlabel("Embedding dimensionality")

    ax.set_ylabel("Mean accuracy (20 runs)")

    ax.set_ylim([0,1.1])

    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
