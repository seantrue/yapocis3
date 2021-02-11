"""
Wrappers for binary operations of the form c = a + b, in both resident and inout form.
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import yapocis.rpc.programs

program, _ = yapocis.rpc.programs.import_program("operators", import_kernels="*", operators=[("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")])
