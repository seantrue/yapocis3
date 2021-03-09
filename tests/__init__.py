#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
import inspect
import os
import string
import time
from random import shuffle
import pytest
import numpy as np

from yapocis.utils.basic import imread, show_array

ASIZE=1200

def image_path_test(image_name="test.jpg"):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, image_name)

def image_test():
    test_image = image_path_test()
    img = imread(test_image).astype(np.float32) / 256
    return img

def text_test_lines(text_name="test.txt"):
    dirname = os.path.dirname(__file__)
    lines = [line.strip() for line in open(os.path.join(dirname, "test.txt")).read().splitlines()]
    return lines

def text_test():
    dirname = os.path.dirname(__file__)
    s = open(os.path.join(dirname, "test.txt")).read()
    uninteresting = set(string.printable) - set(string.ascii_lowercase+string.ascii_uppercase)
    translator=s.maketrans({c:" " for c in uninteresting})
    s=s.translate(translator)
    words = [word for word in set(s.split()) if len(word) > 2]
    return words


def interesting_words(n=10):
    words = text_test()
    shuffle(words)
    return words[:n]


def double_shuffle(lst, n):
    while len(lst) < n:
        lst += lst
    lst = lst[:n]
    shuffle(lst)
    return lst

@pytest.fixture(scope="session", autouse=True)
def show_snapshots(request):
    from yapocis.rpc.device_manager import display_snapshots
    request.addfinalizer(display_snapshots)

def strip_test_prefix(s):
    if s.startswith("test_"):
        s = s[5:]
    return s

def name_of_test(depth=1):
    current_frame = inspect.currentframe()
    call_frames = inspect.getouterframes(current_frame, depth+1)
    fi = call_frames[depth]
    caller_name = fi.function
    file_name = fi.filename
    file_name = os.path.splitext(os.path.basename(os.path.split(file_name)[-1]))[0]
    return strip_test_prefix(caller_name), strip_test_prefix(file_name)


def show(*args):
    tname, fname = name_of_test(2)
    show_array(f"{fname}: {tname}", *args)

def snapshot(snapper):
    tname, fname = name_of_test(2)
    snapper.snapshot(name=f"{fname}/{tname}")

def snapshow(snapper, *args, depth=2):
    tname, fname = name_of_test(depth)
    snapper.snapshot(name=f"{fname}/{tname}")
    show_array(f"{fname}: {tname}", *args)


class TS:
    label = "anisodiff"
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    def __enter__(self):
        self.t = time.time()
        return self
    def __call__(self, snapper, *args):
        t = time.time()-self.t
        snapshow(snapper, f"{t:.03f}sec", *self.args, *args,depth=3)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return