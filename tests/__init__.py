#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os
import numpy as np
from yapocis.utils.basic import imread

def image_path_test(image_name="test.jpg"):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, image_name)

def image_test():
    test_image = image_path_test()
    img = imread(test_image).astype(np.float32) / 256
    return img