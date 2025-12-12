from ectorch.core import population
import torch
from ectorch.config import config
import sys

def score_deb(population:"Population")->torch.Tensor:
    """Calculates the score using Deb's constraint handling method.

    Args:
        population (Population): The population to be scored.
    Returns:
        torch.Tensor: The score tensor of shape (population_size,).
    Note:
        - This method assumes single-objective optimization.
    """
    penalty = population.penalty # Shape : (population_size, num_penalties)
    fitness = population.fitness # Shape : (population_size, num_objectives)
    assert fitness.size(1) == 1, "Deb's method is only applicable for single-objective optimization."

    # Check if there are any constraints
    if penalty is None:
        # Since lower fitness is better and higher sore is better, we negate the fitness
        return -fitness[:, 0]
    else:
        return -fitness[:,0] -torch.sum(penalty, dim=1)


def is_dominated(f1:torch.Tensor, f2:torch.Tensor)->bool:
    """Checks if fitness f1 is dominated by fitness f2.

    Args:
        f1 (torch.Tensor): Fitness tensor of individual 1 of shape (num_objectives,).
        f2 (torch.Tensor): Fitness tensor of individual 2 of shape (num_objectives,).
    Returns:
        bool: True if f1 is dominated by f2, else False.
    Note:
        - This method assumes multi-objective optimization.
        - This method doesnt consider constraints.
    """
    mask1 = torch.prod(f2 <= f1)
    mask2 = torch.sum(f2 < f1)
    return (mask1*mask2).bool()


def get_dominated_map(fitness:torch.Tensor)->torch.Tensor:
    """Generates a dominated map D for the population.

    D(i,j) = True if individual i is dominated by individual j, else False.

    Args:
        fitness (torch.Tensor): The fitness tensor of the population.
    Returns:
        torch.Tensor: A boolean tensor of shape (population_size, population_size) indicating whether each individual is dominated.
    Note:
        - This method assumes multi-objective optimization.
        - This method doesnt consider constraints.
    """
    is_dominated_vmap = torch.vmap(is_dominated, in_dims=(None, 0), out_dims=0)
    D = torch.stack([is_dominated_vmap(f, fitness) for f in fitness], dim=0)
    return D

def get_rank(fitness:torch.Tensor)->torch.Tensor:
    """Calculates the rank of each individual in the population based on domination.

    This score was proposed in MOGA.
    Rank is defined as 1 + the number of individuals that dominate a given individual.

    Args:
        fitness (torch.Tensor): The fitness tensor of the population.
    Returns:
        torch.Tensor: A tensor of shape (population_size,) containing the rank of each individual.
    Note:
        - This method assumes multi-objective optimization.
        - This method doesnt consider constraints.
    """
    dominated_map = get_dominated_map(fitness)
    rank = 1. + torch.sum(dominated_map, dim=1)
    return rank

def get_pareto(fitness:torch.Tensor)->torch.Tensor:
    """Identifies Pareto-optimal individuals in the population.

    Args:
        fitness (torch.Tensor): The fitness tensor of the population. Shape is (population_size, num_objectives).
    Returns:
        torch.Tensor: A tensor of indices of Pareto-optimal individuals.
    Note:
        - This method assumes multi-objective optimization.
        - This method doesnt consider constraints.
    """
    dominated_map = get_dominated_map(fitness)
    is_pareto = torch.sum(dominated_map, dim=1) == 0
    pareto_indices = torch.nonzero(is_pareto, as_tuple=False).squeeze(1)
    return pareto_indices

def get_frontrank(fitness:torch.Tensor)->torch.Tensor:
    """Calculates the front rank F of each individual in the population.

    Front rank was proposed in NSGA algorithm.
    - If front rank is 1, the individual is non-dominated.
    - If front rank is 2, the individual is not dominated by others except those in front 1.
    - If front rank is k, the individual is not dominated by others except those in fronts 1 to k-1.

    Args:
        fitness (torch.Tensor): The fitness tensor of the population. Shape is (population_size, num_objectives).
    Returns:
        torch.Tensor: A tensor of shape (population_size,) containing the front rank of each individual.
    Note:
        - This method assumes multi-objective optimization.
        - This method doesnt consider constraints.
    """
    population_size = fitness.size(0)
    dominated_map = get_dominated_map(fitness)
    front_rank = torch.zeros(population_size, device=fitness.device)
    current_rank = 1

    unranked_indices = torch.arange(population_size, device=fitness.device)
    while unranked_indices.numel() > 0:
        dominated_submap = dominated_map[unranked_indices][:, unranked_indices]
        non_dominated_mask = torch.sum(dominated_submap, dim=1) == 0
        non_dominated_indices = unranked_indices[non_dominated_mask]

        front_rank[non_dominated_indices] = current_rank

        unranked_indices = unranked_indices[~non_dominated_mask]
        current_rank += 1
    
    return front_rank

def get_crowding_distance(fitness:torch.Tensor)->torch.Tensor:
    """Calculates the crowding distance for each individual in the population.

    Crowding distance was proposed in NSGA-II algorithm.
    - Distance is infinite for boundary individuals.
    - Distance is calculated between individuals of the same front.

    Args:
        fitness (torch.Tensor): The fitness tensor of the population. Shape is (population_size, num_objectives).
    Returns:
        torch.Tensor: A tensor of shape (population_size,) containing the crowding distance of each individual.
    Note:
        - This method assumes multi-objective optimization.
        - This method doesnt consider constraints.
    """
    population_size, num_objectives = fitness.size()
    crowding_dist = torch.zeros(population_size, device=fitness.device)
    front_rank = get_frontrank(fitness)
    max_dist = torch.finfo(torch.float32).max

    for rank in torch.unique(front_rank):
        front_indices = torch.nonzero(front_rank == rank, as_tuple=False).squeeze(1)
        front_fitness = fitness[front_indices]

        for obj_idx in range(num_objectives):
            sorted_indices = torch.argsort(front_fitness[:, obj_idx])
            sorted_fitness = front_fitness[sorted_indices, obj_idx]

            crowding_dist[front_indices[sorted_indices[0]]] = max_dist
            crowding_dist[front_indices[sorted_indices[-1]]] = max_dist

            fitness_range = sorted_fitness[-1] - sorted_fitness[0]
            crowding_dist[front_indices[sorted_indices[1:-1]]] = (sorted_fitness[2:] - sorted_fitness[:-2]) / (fitness_range + 1e-10)
    
    return crowding_dist