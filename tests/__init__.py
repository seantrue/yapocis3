#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os

def test_image_path(image_name="test.jpg"):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, image_name)
