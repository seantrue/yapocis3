from yapocis.rpc import interfaces, kernels

program = kernels.import_program("operators", operators=[("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")])
