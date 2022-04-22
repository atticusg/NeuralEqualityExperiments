from datasets import EqualityDataset, PremackDataset
import numpy as np
import pytest
from trained_datasets import TrainedEqualityDataset, TrainedPremackDataset
import utils


@pytest.fixture
def tiny_equality_dataset():
    return EqualityDataset(embed_dim=10, n_pos=2, n_neg=10, flatten=True)


@pytest.fixture
def tiny_trained_equality_dataset():
    return TrainedEqualityDataset(embed_dim=10, n_pos=2, n_neg=10, flatten=True)


def test_equality_create_pos(tiny_equality_dataset):
    result = tiny_equality_dataset._create_pos()
    assert len(result) == tiny_equality_dataset.n_pos
    for (v1, v2), label in result:
        assert label == tiny_equality_dataset.POS_LABEL
        assert np.array_equal(v1, v2)


def test_equality_create_neg(tiny_equality_dataset):
    result = tiny_equality_dataset._create_neg()
    assert len(result) == tiny_equality_dataset.n_neg
    for (v1, v2), label in result:
        assert label == tiny_equality_dataset.NEG_LABEL
        assert not np.array_equal(v1, v2)


@pytest.mark.parametrize("dataset_class, flatten, expected", [
    [EqualityDataset, True, (4, 20)],
    [TrainedEqualityDataset, False, (4, 2, 10)]
])
def test_flatten(dataset_class, flatten, expected):
    dataset = dataset_class(embed_dim=10, n_pos=2, n_neg=2, flatten=flatten)
    assert dataset.flatten == flatten
    X, y = dataset.create()
    result = X.shape
    assert result == expected


@pytest.mark.parametrize("dataset_class, cls, expected", [
    [EqualityDataset, 1, 2],
    [TrainedEqualityDataset, 0, 10]
])
def test_equality_create_label_dist(dataset_class, cls, expected):
    dataset = dataset_class(embed_dim=2, n_pos=2, n_neg=10)
    X, y = dataset.create()
    result = sum([1 for label in y if label == cls])
    assert result == expected


@pytest.mark.parametrize("dataset_class, cls, expected", [
    [EqualityDataset, 1, True],
    [TrainedEqualityDataset, 0, False]
])
def test_equality_create_vector_relations(dataset_class, cls, expected):
    dataset = dataset_class(embed_dim=2, n_pos=2, n_neg=2, flatten=False)
    dataset.create()
    for (v1, v2), label in dataset.data:
        if label == cls:
            rel = np.array_equal(v1, v2)
            assert rel == expected


@pytest.mark.parametrize("dataset_class, flatten", [
    [EqualityDataset, True],
    [EqualityDataset, False],
    [TrainedEqualityDataset, True],
    [TrainedEqualityDataset, False]
])
def test_equality_disjoint(dataset_class, flatten):
    dataset = dataset_class(embed_dim=2, n_pos=2, n_neg=2, flatten=flatten)
    dataset.create()
    with pytest.raises(AssertionError):
        dataset.test_disjoint(dataset)


@pytest.mark.parametrize("dataset_class", [
    PremackDataset,
    TrainedPremackDataset
])
def test_premack_create_same_same(dataset_class):
    n_pos = 20
    dataset = dataset_class(n_pos=n_pos, flatten_root=True, flatten_leaves=False)
    if 'Trained' in dataset_class.__name__:
        examples = [utils.randvec(10) for _ in range(n_pos)]
        result = dataset._create_same_same(examples)
    else:
        result = dataset._create_same_same()
    assert len(result) == dataset.n_same_same
    for (p1, p2), label in result:
        assert label == dataset.POS_LABEL
        assert np.array_equal(p1[0], p1[1])
        assert np.array_equal(p2[0], p2[1])


@pytest.mark.parametrize("dataset_class", [
    PremackDataset,
    TrainedPremackDataset
])
def test_premack_create_diff_diff(dataset_class):
    n_pos = 20
    dataset = dataset_class(n_pos=n_pos, flatten_root=True, flatten_leaves=False)
    if 'Trained' in dataset_class.__name__:
        examples = [utils.randvec(10) for _ in range(n_pos * 2)]
        result = dataset._create_diff_diff(examples)
    else:
        result = dataset._create_diff_diff()
    assert len(result) == dataset.n_diff_diff
    for (p1, p2), label in result:
        assert label == dataset.POS_LABEL
        assert not np.array_equal(p1[0], p1[1])
        assert not np.array_equal(p2[0], p2[1])


@pytest.mark.parametrize("dataset_class", [
    PremackDataset,
    TrainedPremackDataset
])
def test_premack_create_same_diff(dataset_class):
    n_neg = 20
    vecs_needed = 30
    dataset = dataset_class(n_neg=n_neg, flatten_root=True, flatten_leaves=False)
    if 'Trained' in dataset_class.__name__:
        examples = [utils.randvec(10) for _ in range(vecs_needed)]
        result = dataset._create_same_diff(examples)
    else:
        result = dataset._create_same_diff()
    assert len(result) == dataset.n_same_diff
    for (p1, p2), label in result:
        assert label == dataset.NEG_LABEL
        assert np.array_equal(p1[0], p1[1])
        assert not np.array_equal(p2[0], p2[1])


@pytest.mark.parametrize("dataset_class", [
    PremackDataset,
    TrainedPremackDataset
])
def test_premack_create_diff_same(dataset_class):
    n_neg = 20
    vecs_needed = 30
    dataset = dataset_class(n_neg=n_neg, flatten_root=True, flatten_leaves=False)
    if 'Trained' in dataset_class.__name__:
        examples = [utils.randvec(10) for _ in range(vecs_needed)]
        result = dataset._create_diff_same(examples)
    else:
        result = dataset._create_diff_same()
    assert len(result) == dataset.n_diff_same
    for (p1, p2), label in result:
        assert label == dataset.NEG_LABEL
        assert not np.array_equal(p1[0], p1[1])
        assert np.array_equal(p2[0], p2[1])


@pytest.mark.parametrize("dataset_class, flatten_root, flatten_leaves", [
    [PremackDataset, True, True],
    [PremackDataset, True, False],
    [PremackDataset, False, True],
    [PremackDataset, False, False],
    [TrainedPremackDataset, True, True],
    [TrainedPremackDataset, True, False],
    [TrainedPremackDataset, False, True],
    [TrainedPremackDataset, False, False]
])
def test_premack_disjoint(dataset_class, flatten_root, flatten_leaves):
    dataset = dataset_class(
        embed_dim=2, n_pos=40, n_neg=40,
        flatten_root=flatten_root,
        flatten_leaves=flatten_leaves)
    dataset.create()
    with pytest.raises(AssertionError):
        dataset.test_disjoint(dataset)


@pytest.mark.parametrize("dataset_class, flatten_root, flatten_leaves, expected", [
    [PremackDataset, True, True, (80, 40)],
    [PremackDataset, True, False, (80, 40)],
    [PremackDataset, False, True, (80, 2, 20)],
    [PremackDataset, False, False, (80, 2, 2, 10)],
    [TrainedPremackDataset, True, True, (80, 40)],
    [TrainedPremackDataset, True, False, (80, 40)],
    [TrainedPremackDataset, False, True, (80, 2, 20)],
    [TrainedPremackDataset, False, False, (80, 2, 2, 10)]
])
def test_premack_flattening(dataset_class, flatten_root, flatten_leaves, expected):
    dataset = dataset_class(
        embed_dim=10, n_pos=40, n_neg=40,
        flatten_root=flatten_root,
        flatten_leaves=flatten_leaves)
    assert dataset.flatten_root == flatten_root
    assert dataset.flatten_leaves == flatten_leaves
    X, y = dataset.create()
    result = X.shape
    assert result == expected


@pytest.mark.parametrize("dataset_class, cls, expected", [
    [PremackDataset, 1, 40],
    [PremackDataset, 0, 80],
    [TrainedPremackDataset, 1, 40],
    [TrainedPremackDataset, 0, 80]
])
def test_premack_create_label_dist(dataset_class, cls, expected):
    dataset = dataset_class(embed_dim=2, n_pos=40, n_neg=80)
    X, y = dataset.create()
    result = sum([1 for label in y if label == cls])
    assert result == expected


@pytest.mark.parametrize("dataset_class, n_pos, n_neg", [
    [PremackDataset, 20, 41],
    [PremackDataset, 41, 20],
    [TrainedPremackDataset, 0, 41],
    [TrainedPremackDataset, 41, 20]
])
def test_premack_odd_size_value_error(dataset_class, n_pos, n_neg):
    with pytest.raises(ValueError):
        dataset_class(embed_dim=2, n_pos=n_pos, n_neg=n_neg)
