#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.images import find_image, images
from yapocis.utils.basic import imread

def test_images():
    tifs = [name for name in images() if name.endswith(".tif")]
    for tif in tifs:
        img = imread(find_image(tif))

