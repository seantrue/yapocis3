#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

#
# interfacecl.py
#
# Process an improper subset of the CORBA IDL grammar to facilitate calling opencl from a python Numpy framework
# Inspired by an IDL parser by Paul McGuire, shipped as a demo with pyparser
#

from yapocis.utils.typing import *
from yapocis.rpc.interface import get_interfaces, python_stubs, opencl_stubs, dump_interface






if __name__ == '__main__':
    s = """
    interface anisodiff {
        // HxW image
        kernel filterImage(width_of input, height_of input, in float input, float scale, float size_step, out_like input);
        alias filterImage_res as filterImage(width_of input, height_of input, resident float input, float scale, float size_step, resident float output);
    };
    interface graphics {
        # Kernels have aliases for running on resident buffers
        kernel lines(width_of a, height_of a, inout float a, width_of lines, in float lines, float distance, in float rcolor);
        alias lines_in as lines(width_of a, height_of a, resident float a, width_of lines, in float lines, float distance, in float rcolor);
    };

    """
    interfaces= get_interfaces(s)
    for interface in interfaces:
        python_stubs(interface)
        opencl_stubs(interface)
        dump_interface(interface)
