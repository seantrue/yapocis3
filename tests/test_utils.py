#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
import numpy as np
from yapocis.utils.stereo_typing import *

from yapocis.utils.basic import chunker
from yapocis.utils.internals import get_project_dir
from yapocis.lib.channels import join_channels, join_planes, split_channels, split_planes


# TODO: Add more tests from utils

def test_chunker():
    assert len(list(chunker([]))) == 0
    assert len(list(chunker([1]))) == 1
    assert len(list(chunker([0,1,2,3],2))) == 2
    assert len(list(chunker([0,1,2],2))) == 2
    assert len(list(chunker([0,1],2))) == 1

def test_split_join():
  a = np.random.uniform(0,1.0,(10,10))
  b = np.random.uniform(0,1.0,(10,10))
  c = np.random.uniform(0,1.0,(10,10))
  abc = join_channels(a,b,c)
  a1,b1,c1 = split_channels(abc)
  assert np.sum(a-a1) == 0.0
  assert np.sum(b-b1) == 0.0
  assert np.sum(c-c1) == 0.0
  abc = join_planes(a,b,c)
  a1,b1,c1 = split_planes(abc)
  assert np.sum(a-a1) == 0.0
  assert np.sum(b-b1) == 0.0
  assert np.sum(c-c1) == 0.0

def test_project_dir():
    assert get_project_dir()