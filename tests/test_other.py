from yapocis.utils import imread, show_array
from yapocis.kernels.hsi import *
import yapocis.kernels.peronamalik2 as peronamalik2

def test_perona_malik():
   rgb = imread("tests/test.jpg").astype(np.float32)/256
   show_array("org", rgb)
   for i, rgb2 in enumerate(peronamalik2.pm_filter_rgbIS(rgb)):
      if i >= 40:
         break
   show_array("Filtered", rgb2)

test_perona_malik()