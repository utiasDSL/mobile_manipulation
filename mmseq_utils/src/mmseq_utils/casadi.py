from casadi.tools import entry, struct_symMX, struct_MX

def casadi_sym_struct_old(data: dict):
    param_entries = []
    for (name, shape) in data.items():
        param_entries.append(entry(name, shape=shape))
    
    return struct_symMX(param_entries)

def casadi_sym_struct(data: dict):
    param_entries = []
    for (name, expression) in data.items():
        param_entries.append(entry(name, expr=expression))
    
    return struct_MX(param_entries)