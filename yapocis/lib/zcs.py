"""
Low level wrapper for zero crossing detector kernel
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import yapocis.rpc.programs
zcs = zcs_res = None
yapocis.rpc.programs.import_program("zcs",import_kernels="*")
