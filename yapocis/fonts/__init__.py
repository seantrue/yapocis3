import os
try:
    from PIL.ImageFont import truetype
except ImportError:
    truetype = None

def fonts():
    dirname = os.path.dirname(__file__)
    fonts = [font_name for font_name in os.listdir(dirname) if font_name.endswith(".ttf")]
    return fonts

def font_path(font_name):
    dirname = os.path.dirname(__file__)
    font_path = os.path.join(dirname, font_name)
    if not os.path.exists(font_path):
        raise ValueError(f"{font_name} not found, use one of: {' '.join(fonts())}")
    return font_path

def font(font_suggestion):
    try:
        return font_path(font_suggestion)
    except ValueError:
        font_names = fonts()
        for font_name in font_names:
            if font_suggestion in font_name:
                return font_path(font_name)
        for font_name in font_names:
            if font_suggestion.lower() in font_name.lower():
                return font_path(font_name)
    return font_path("Ubuntu-Regular.ttf")


_fontcache = {}
def image_font(font_name="default", size=14, index=0, encoding="", layout_engine=None):
    if truetype is None:
        raise RuntimeError(f"PIL with TrueType support is not installed")
    font_path = font(font_name)
    try:
        return _fontcache[(font_path,size,index)]
    except KeyError:
        _fontcache[(font_path,size,index)]  = fnt = truetype(font=font_path, size=size, index=index, encoding=encoding, layout_engine=layout_engine)
        return fnt

