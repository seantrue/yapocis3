import yapocis.rpc.programs
from yapocis.rpc import interfaces, engines

program = yapocis.rpc.programs.import_program("operators", operators=[("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")])
