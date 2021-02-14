#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.utils.basic import chunker

def test_chunker():
    assert len(list(chunker([]))) == 0
    assert len(list(chunker([1]))) == 1
    assert len(list(chunker([0,1,2,3],2))) == 2
    assert len(list(chunker([0,1,2],2))) == 2
    assert len(list(chunker([0,1],2))) == 1
