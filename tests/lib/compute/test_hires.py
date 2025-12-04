#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from yapocis.images import find_image, images
from yapocis.utils.basic import imread, show_array


def test_images():
    tifs = [name for name in images() if name.endswith(".tif")]
    for tif in tifs:
        img = imread(find_image(tif))
        show_array(img, "imread", tif)
