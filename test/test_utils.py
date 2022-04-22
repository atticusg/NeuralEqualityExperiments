import numpy as np
import pytest
import random
import utils


@pytest.mark.parametrize("set_value", [True, False])
def test_fix_random_seeds_system(set_value):
    params = dict(
        seed=42,
        set_system=set_value,
        set_torch=False,
        set_torch_cudnn=False)
    utils.fix_random_seeds(**params)
    x = np.random.random()
    utils.fix_random_seeds(**params)
    y = np.random.random()
    assert (x == y) == set_value


@pytest.mark.parametrize("set_value", [True, False])
def test_fix_random_seeds_pytorch(set_value):
    import torch
    params = dict(
        seed=42,
        set_system=False,
        set_torch=set_value,
        set_torch_cudnn=set_value)
    utils.fix_random_seeds(**params)
    x = torch.rand(1)
    utils.fix_random_seeds(**params)
    y = torch.rand(1)
    assert (x == y) == set_value
