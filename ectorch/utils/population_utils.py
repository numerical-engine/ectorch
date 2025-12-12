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

def split(population:'Population', indices:list[int] | int, shuffle = True)->list['Population']:
    """Split a Population into multiple Populations based on the given indices.

    Args:
        population (Population): The Population instance to split.
        indices (list[int] | int): A list of indices or a single integer indicating the split points.
            - If a list of integers is provided, it indicates the end indices for each split.
            - If a single integer is provided, it indicates the size of each split.
        shuffle (bool): Whether to shuffle the population before splitting. Default is True.
    Returns:
        list[Population]: A list of Population instances resulting from the split.
    """
    if shuffle:
        population.shuffle()
    if isinstance(indices, int):
        population_size = len(population)
        assert population_size % indices == 0, "Population size must be divisible by the split size."
        indices = list(range(indices, population_size, indices))
    
    prev_index = 0
    populations = []
    for index in indices:
        selected_indices = torch.arange(prev_index, index, device = population.device)
        populations.append(population.index_select(selected_indices))
        prev_index = index
    
    if prev_index < len(population):
        selected_indices = torch.arange(prev_index, len(population), device = population.device)
        populations.append(population.index_select(selected_indices))

    return populations

def uniform_samplingR(xl:torch.Tensor, xu:torch.Tensor, population_size:int)->torch.Tensor:
    """Generates uniform random samples of real number within given bounds.

    Args:
        xl (torch.Tensor): A tensor of shape (variable_dimension,) representing the lower bounds.
        xu (torch.Tensor): A tensor of shape (variable_dimension,) representing the upper bounds.
        population_size (int): The number of samples to generate.
    Returns:
        torch.Tensor: A tensor of shape (population_size, variable_dimension) containing the generated samples.
    """
    assert xl.dim() == 1, "Lower bounds tensor xl should be 1-dimensional."
    assert xu.dim() == 1, "Upper bounds tensor xu should be 1-dimensional."
    assert xl.size(0) == xu.size(0), "Lower and upper bounds tensors should have the same dimension."

    variable_dimension = xl.size(0)
    samples = torch.rand(population_size, variable_dimension) * (xu - xl) + xl
    return samples

def uniform_samplingZ(xl:torch.Tensor, xu:torch.Tensor, population_size:int)->torch.Tensor:
    """Generates uniform random samples of integer within given bounds.

    Args:
        xl (torch.Tensor): A tensor of shape (variable_dimension,) representing the lower bounds.
        xu (torch.Tensor): A tensor of shape (variable_dimension,) representing the upper bounds.
        population_size (int): The number of samples to generate.
    Returns:
        torch.Tensor: A tensor of shape (population_size, variable_dimension) containing the generated samples.
    """
    assert xl.dim() == 1, "Lower bounds tensor xl should be 1-dimensional."
    assert xu.dim() == 1, "Upper bounds tensor xu should be 1-dimensional."
    assert xl.size(0) == xu.size(0), "Lower and upper bounds tensors should have the same dimension."

    variable_dimension = xl.size(0)
    samples = torch.stack(
        [torch.randint(low=xl[i].item(), high=xu[i].item() + 1, size=(population_size,), dtype=torch.float32) for i in range(variable_dimension)],
        dim=1
    )
    return samples

def uniform_samplingB(population_size:int, variable_dimension:int)->torch.Tensor:
    """Generates uniform random samples of binary values.

    Args:
        population_size (int): The number of samples to generate.
        variable_dimension (int): The dimension of each sample.
    Returns:
        torch.Tensor: A tensor of shape (population_size, variable_dimension) containing the generated samples.
    """
    samples = torch.randint(0, 2, (population_size, variable_dimension), dtype=torch.float32)
    return samples


def get_shared_value(points:torch.Tensor, threshold:float, alpha:float)->torch.Tensor:
    """Get shared function value Sh (2D tensor) defined by

    Sh(i,j) = 1 - (d(i,j)/threshold)^alpha if d(i,j) < threshold else 0. d(i,j) is the Euclidean distance between point i and point j.

    Args:
        points (torch.Tensor): A tensor of shape (num_points, num_dimensions) representing the coordinates of points.
        threshold (float): The distance threshold for sharing.
        alpha (float): The exponent parameter for sharing.
    Returns:
        torch.Tensor: A tensor of shape (num_points, num_points) representing the shared function values.
    """
    num_points = points.size(0)
    dist_matrix = torch.cdist(points, points, p=2)  # Shape: (num_points, num_points)
    
    shared_value = torch.zeros((num_points, num_points), device=points.device)
    mask = dist_matrix < threshold
    shared_value[mask] = 1.0 - (dist_matrix[mask] / threshold) ** alpha
    
    return shared_value