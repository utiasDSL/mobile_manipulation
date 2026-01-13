from casadi.tools import entry, struct_MX


def casadi_sym_struct(data: dict):
    """Create CasADi symbolic structure from dictionary.

    Args:
        data (dict): Dictionary mapping parameter names to CasADi symbolic expressions.

    Returns:
        casadi.struct_MX: CasADi structure containing the symbolic parameters.
    """
    param_entries = []
    order = []

    for name, expression in data.items():
        param_entries.append(entry(name, expr=expression))
        order.append(name)

    return struct_MX(param_entries, order)


def reconstruct_sym_struct_map_from_array(p_struct, p_map_array):
    """Reconstruct CasADi structure map from flattened array.

    Args:
        p_struct (casadi.struct_MX): CasADi structure template.
        p_map_array (ndarray): Flattened parameter array.

    Returns:
        casadi.struct_MX: Reconstructed structure map with reshaped parameters.
    """
    p_map_new = p_struct(0)
    index = 0
    for key in p_struct.order:
        shape = p_struct[key].shape
        size = shape[0] * shape[1]

        p_map_new[key] = p_map_array[index : index + size].reshape(shape).T

        index += size

    return p_map_new
