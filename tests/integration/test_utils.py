#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
import os

import numpy as np

from yapocis.lib.channels import (
    join_channels,
    join_planes,
    split_channels,
    split_planes,
)
from yapocis.utils.basic import chunker
from yapocis.utils.internals import get_project_dir
from yapocis.utils.stereo_typing import *

# TODO: Add more tests from utils


def test_chunker():
    assert len(list(chunker([]))) == 0
    assert len(list(chunker([1]))) == 1
    assert len(list(chunker([0, 1, 2, 3], 2))) == 2
    assert len(list(chunker([0, 1, 2], 2))) == 2
    assert len(list(chunker([0, 1], 2))) == 1


def test_split_join():
    a = np.random.uniform(0, 1.0, (10, 10))
    b = np.random.uniform(0, 1.0, (10, 10))
    c = np.random.uniform(0, 1.0, (10, 10))
    abc = join_channels(a, b, c)
    parts = split_channels(abc)
    # a tuple, as annotated: callers that take several results by type (ImpCraft's function
    # nodes) read a list as one value
    assert isinstance(parts, tuple) and len(parts) == 3
    a1, b1, c1 = parts
    assert np.sum(a - a1) == 0.0
    assert np.sum(b - b1) == 0.0
    assert np.sum(c - c1) == 0.0
    abc = join_planes(a, b, c)
    a1, b1, c1 = split_planes(abc)
    assert np.sum(a - a1) == 0.0
    assert np.sum(b - b1) == 0.0
    assert np.sum(c - c1) == 0.0


def test_project_dir():
    project_dir = get_project_dir()
    assert project_dir is None or os.path.exists(project_dir)
