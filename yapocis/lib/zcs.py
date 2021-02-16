"""
Low level wrapper for zero crossing detector kernel
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.rpc.programs import import_program

zcs1 = zcs1_res = zcs2 = zcs2_res = zcs3 = zcs3_res = zcs4 = zcs4_res = None
program, kernels = import_program("zcs", import_kernels="*")


def zcs(image):
    if len(image.shape) == 2:
        handler = zcs1
        global_size = image.size
    elif image.shape[2] == 2:
        handler = zcs2
        global_size = image.size // 2
    elif image.shape[2] == 3:
        handler = zcs3
        global_size = image.size // 3
    elif image.shape[2] == 4:
        handler = zcs4
        global_size = image.size // 4
    return handler(image, global_size=global_size)


def zcs_res(image, zcs):
    if len(image.shape) == 2:
        handler = zcs1_res
        global_size = image.size
    elif image.shape[2] == 2:
        handler = zcs2_res
        global_size = image.size // 2
    elif image.shape[2] == 3:
        handler = zcs3_res
        global_size = image.size // 3
    elif image.shape[2] == 4:
        handler = zcs4_res
        global_size = image.size // 4
    handler(image, zcs, global_size=global_size)
