'''
Created on Jul 22, 2011

@author: seant
'''
import yapocis.rpc.programs

DEBUG = False
from yapocis.rpc import engines, interfaces
from yapocis.yapocis_types import *

rgb2hsi_cl = hsi2rgb_cl = None
yapocis.rpc.programs.import_program("hsi", rgb2hsi="rgb2hsi_cl", hsi2rgb="hsi2rgb_cl")

def rgb2hsi(r:Array, g:Array, b:Array) -> Tuple[Array,Array,Array]:
    h, s, i = rgb2hsi_cl(r, g, b) #@UnusedVariable
    return h, s, i

def hsi2rgb(h:Array, s:Array, i:Array) -> Tuple[Array,Array,Array]:
    r,g,b = hsi2rgb_cl(h,s,i)
    return r,g,b


