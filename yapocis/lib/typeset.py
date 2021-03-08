# -*- coding: utf-8 -*-
# Adapted from wordle.py from freetype-py and used under the following terms:
# -----------------------------------------------------------------------------
#
#  FreeType high-level python API - Copyright 2011-2015 Nicolas P. Rougier
#  Distributed under the terms of the new BSD license.
#
# -----------------------------------------------------------------------------
import math
import numpy as np
from freetype import FT_Matrix, FT_LOAD_RENDER, FT_Vector, FT_Set_Transform, byref, Face
from yapocis.fonts import font
from yapocis.utils.typing import Optional, Image1, Tuple
from yapocis.utils.basic import set_margin

# These are performance boosts excepwhen size, font, or rotation are changing randomly
_glyph_cache = {}
_kerning_cache = {}


def render_freetype(text, filename: str, size: float = 12, angle: float = 0):
    """
    Parameters:
    -----------
    text : string
        Text to be displayed
    filename : string
        Path to a font
    size : int
        Font size in points
    angle : float
        Text angle in degrees
    """
    angle = round(angle, 4)
    face = Face(filename)
    face.set_char_size(int(size * 64))
    angle = (angle / 180.0) * math.pi
    matrix = FT_Matrix((int)(math.cos(angle) * 0x10000),
                       (int)(-math.sin(angle) * 0x10000),
                       (int)(math.sin(angle) * 0x10000),
                       (int)(math.cos(angle) * 0x10000))
    flags = FT_LOAD_RENDER
    pen = FT_Vector(0, 0)
    FT_Set_Transform(face._FT_Face, byref(matrix), byref(pen))
    previous = 0
    xmin, xmax = 0, 0
    ymin, ymax = 0, 0
    for c in text:
        try:
            kerning_x, left, rows, top, width, advance_x, advance_y = _kerning_cache[
                (filename, size, angle, previous, c)]
        except KeyError:
            face.load_char(c, flags)
            kerning_x = face.get_kerning(previous, c).x
            previous = c
            glyph = face.glyph
            advance_x = glyph.advance.x
            advance_y = glyph.advance.y
            width = glyph.bitmap.width
            rows = glyph.bitmap.rows
            top = glyph.bitmap_top
            left = glyph.bitmap_left
            _kerning_cache[
                (filename, size, angle, previous, c)] = kerning_x, left, rows, top, width, advance_x, advance_y
        previous = c
        pen.x += kerning_x
        x0 = (pen.x >> 6) + left
        x1 = x0 + width
        y0 = (pen.y >> 6) - (rows - top)
        y1 = y0 + rows
        xmin, xmax = min(xmin, x0), max(xmax, x1)
        ymin, ymax = min(ymin, y0), max(ymax, y1)
        pen.x += advance_x
        pen.y += advance_y

    L = np.zeros((ymax - ymin, xmax - xmin), dtype=np.ubyte)
    previous = 0
    pen.x, pen.y = 0, 0
    for c in text:
        face.load_char(c, flags)
        kerning = face.get_kerning(previous, c)
        previous = c
        glyph = face.glyph
        bitmap = face.glyph.bitmap
        pitch = bitmap.pitch
        width = bitmap.width
        rows = bitmap.rows
        top = glyph.bitmap_top
        left = glyph.bitmap_left
        pen.x += kerning.x
        x = (pen.x >> 6) - xmin + left
        y = (pen.y >> 6) - ymin - (rows - top)
        try:
            data = _glyph_cache[(filename, size, angle, c)]
        except KeyError:
            data = np.empty(rows * width, dtype=np.ubyte)
            buffer = bitmap.buffer.copy()
            for j in range(rows):
                data[j * width:j * width + width] = buffer[j * pitch:j * pitch + width]
            _glyph_cache[(filename, size, angle, c)] = data
        if len(data):
            Z = data.reshape(rows, width)
            L[y:y + rows, x:x + width] |= Z[::-1, ::1]
        pen.x += face.glyph.advance.x
        pen.y += face.glyph.advance.y

    return L


def render_text(text: str, font_name: str = "default", size: int = 32, rotation: Optional[float] = None,
                debug_bbox=False) \
        -> Tuple[Tuple, Image1]:
    rotation = rotation % 360
    if rotation > 180:
        rotation = -180 + (rotation - 180)
    font_path = font(font_name)
    size = 32 if size is None else size
    rotation = 0.0 if rotation is None else rotation
    rendered = render_freetype(text, font_path, size, rotation)
    rendered = rendered[::-1, :]
    vprofile = np.nonzero(np.sum(rendered, 0))[0]
    hprofile = np.nonzero(np.sum(rendered, 1))[0]
    y1, y2 = hprofile[0], hprofile[-1]
    x1, x2 = vprofile[0], vprofile[-1]
    rendered = rendered[max(0, y1 - 1):y2 + 2, max(0, x1 - 1):x2 + 2]
    if debug_bbox:
        set_margin(rendered, 2, 255)
    return [0, 0, *rendered.shape], rendered.astype(np.float32) / 255.0


def compute_offset(angle, h, w):
    dx = dy = 0
    if angle == 0:
        dy = -h // 2
    elif 0 < angle < 90:
        dy = -h
    elif angle == 90:
        dx = -w // 2
        dy = -h
    elif 90 <= angle < 180:
        dx = -w
        dy = -h
    elif angle == 180 or angle == -180:
        dx = -w
        dy = -h // 2
    elif -90 < angle < 0:
        pass
    elif angle == -90:
        dx = -w / 2
    elif -180 < angle < -90:
        dx = -w
    return dx, dy
