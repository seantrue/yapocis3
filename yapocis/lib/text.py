"""
Support for rendering text into sprites for use with compositing kernels
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np
from PIL import Image,ImageDraw
from yapocis.fonts import image_font
from yapocis.utils.typing import *

_rendered = {}
_render_buffer = None
def get_render_buffer(size:Tuple[int,int]) -> PILImage:
    global _render_buffer
    if _render_buffer is None or _render_buffer.size[0] < size[0] or _render_buffer.size[1] < size[1]:
        _render_buffer = Image.new("L", (size[0],size[1]))
    return _render_buffer

def render_text(text:str, font_name:str="default", size:int=32, rotation:Optional[float]=None) -> Tuple[Tuple, Image1]:
    try:
        bbox, a = _rendered[(text,font_name,size, rotation)]
        return bbox, a
    except KeyError:
        pass
    font = image_font(font_name, size=size)
    bbox = font.getbbox(text)
    x,y, w,h = bbox
    h += 1
    rendered = get_render_buffer((w,h))
    rendered.paste(0, box=(0,0,w,h))
    draw = ImageDraw.ImageDraw(rendered)
    draw.text((0,0),text, font=font,fill=255,stroke=255)
    if rotation:
        pixels =  rendered.crop((0,0,w,h))
        wh = max(w,h)*2
        rendered = get_render_buffer((wh,wh))
        rendered.paste(0, box=(0, 0, wh, wh))
        rendered.paste(pixels,(w//2,wh//2))
        rendered = rendered.rotate(rotation,fillcolor=0)
    a = np.array(rendered,dtype=np.float32)
    vprofile = np.nonzero(np.sum(a,0))[0]
    hprofile = np.nonzero(np.sum(a,1))[0]
    y1,y2 = hprofile[0], hprofile[-1]
    x1,x2 = vprofile[0],vprofile[-1]
    a = a[max(0,y1-1):y2+2, max(0,x1-1):x2+2]
    a /= 255.0
    a = a[:,:]
    _rendered[(text,font_name, size, rotation)] = (bbox,a)
    return bbox, a

