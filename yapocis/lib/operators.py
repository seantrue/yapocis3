"""
Wrappers for binary operations of the form c = a + b, in both resident and inout form.
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import yapocis.rpc.programs

add = add_res = sub = sub_res = mul = mul_res = div = div_res = None
program, _ = yapocis.rpc.programs.import_program("operators", import_kernels="*",
                                                 operators=[("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")])
