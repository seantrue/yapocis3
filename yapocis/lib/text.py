#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np
from PIL import Image,ImageDraw
from yapocis.fonts import image_font
from yapocis.basic import from_image

_rendered = {}
_render_buffer = None
def get_render_buffer(size):
    global _render_buffer
    if _render_buffer is None or _render_buffer.size[0] < size[0] or _render_buffer.size[1] < size[1]:
        _render_buffer = Image.new("L", (size[0],size[1]))
    return _render_buffer

def render_text(text, font_name="default", size=32):
    try:
        a = _rendered[(text,font_name,size)]
        return a
    except KeyError:
        pass
    font = image_font(font_name, size=size)
    bbox = font.getbbox(text)
    x,y, w,h = bbox
    h += 1
    rendered = get_render_buffer((w,h))
    rendered.paste(0, box=(0,0,w,h))
    draw = ImageDraw.ImageDraw(rendered)
    draw.text((0,0),text, font=font,fill=255)
    a = np.array(rendered,dtype=np.float32)
    a /= 255.0
    a = a[y:h+1,:w]
    _rendered[(text,font_name, size)] = a[:,:]
    return a

