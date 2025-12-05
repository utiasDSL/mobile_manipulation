from casadi.tools import entry, struct_MX, struct_symMX


def casadi_sym_struct_old(data: dict):
    param_entries = []
    for name, shape in data.items():
        param_entries.append(entry(name, shape=shape))

    return struct_symMX(param_entries)


def casadi_sym_struct(data: dict):
    param_entries = []
    order = []

    for name, expression in data.items():
        param_entries.append(entry(name, expr=expression))
        order.append(name)

    return struct_MX(param_entries, order)


def reconstruct_sym_struct_map_from_array(p_struct, p_map_array):
    p_map_new = p_struct(0)
    index = 0
    for key in p_struct.order:
        shape = p_struct[key].shape
        size = shape[0] * shape[1]

        p_map_new[key] = p_map_array[index : index + size].reshape(shape).T

        index += size

    return p_map_new


if __name__ == "__main__":
    import casadi as cs
    import numpy as np

    data = {"r_EEPos3": cs.MX.sym("r", 2), "W_EEPos3": cs.MX.sym("W", 3, 3)}
    p_struct = casadi_sym_struct(data)
    print(p_struct.getLabel)
    print(p_struct.labels)
    print(p_struct.keys())
    print(p_struct.order)
    print(p_struct["W_EEPos3"].shape)

    p_map = p_struct(0)
    p_map["r_EEPos3"] = np.arange(2)
    p_map["W_EEPos3"] = np.arange(9).reshape((3, 3))
    p_map_array = p_map.cat.full().flatten()

    p_map_new = p_struct(0)
    reconstruct_sym_struct_map_from_array(p_map_new, p_map_array)
    print(p_map_new.cat.full().flatten())
