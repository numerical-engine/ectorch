import torch
from ectorch.config import config
import sys

def check_variables(variables:dict[str, torch.Tensor])->bool:
    """Checks if the variables are valid.

    Args:
        variables (dict[str, torch.Tensor]): A dictionary representing the variables of the population.
            - Each key is the variable type which should be one of config.var_keys.
            - Each value is a tensor of shape (population_size, variable_dimension) representing the values of the variable.
    Returns:
        bool: True if all variables are valid, False otherwise.
    """
    var_types = set()
    population_size = None
    for var_type, value in variables.items():
        if var_type not in config.var_keys:
            print(f"Invalid variable type: {var_type}. It should be one of {config.var_keys}.", file=sys.stderr)
            return False
        if var_type in var_types:
            print(f"Duplicate variable type: {var_type}. Each variable type should be unique.", file=sys.stderr)
            return False
        var_types.add(var_type)
        if not isinstance(value, torch.Tensor):
            print(f"Variable value for type {var_type} should be a torch.Tensor.", file=sys.stderr)
            return False
        if value.dim() != 2:
            print(f"Variable value for type {var_type} should be a 2D tensor of shape (population_size, variable_dimension).", file=sys.stderr)
            return False
        if population_size is None:
            population_size = value.size(0)
        elif population_size != value.size(0):
            print(f"All variable values should have the same population size. Mismatch found in variable type {var_type}.", file=sys.stderr)
            return False
    return True