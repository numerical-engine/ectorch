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


def cat(population_list:list['Population'])->'Population':
    """Concatenates a list of Population instances into a single Population.

    Args:
        population_list (list[Population]): A list of Population instances to concatenate.
    Returns:
        Population: A new Population instance containing all individuals from the input populations.
    """
    assert len(population_list) > 0, "The population list should not be empty."

    new_generation = max(pop.generation for pop in population_list)
    
    variables = {}
    for var_type in population_list[0].variables.keys():
        variables[var_type] = torch.cat([pop.variables[var_type] for pop in population_list], dim=0)
    
    age = torch.cat([pop.age for pop in population_list], dim=0)
    
    fitness = None
    if population_list[0].fitness is not None:
        fitness = torch.cat([pop.fitness for pop in population_list], dim=0)
    
    penalty = None
    if population_list[0].penalty is not None:
        penalty = torch.cat([pop.penalty for pop in population_list], dim=0)
    
    score = None
    if population_list[0].score is not None:
        score = torch.cat([pop.score for pop in population_list], dim=0)
    
    return type(population_list[0])(
        variables=variables,
        generation=new_generation,
        age=age,
        fitness=fitness,
        penalty=penalty,
        score=score,
    )