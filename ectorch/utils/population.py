import torch
from ectorch.core.population import Population


def squeeze(population:Population, indices:torch.Tensor)->Population:
    """Squeeze the population to keep only the individuals at the specified indices.

    Args:
        population (Population): The original population.
        indices (torch.Tensor): The indices of the individuals to keep. Shape is (M,), where M is the number of individuals to keep.
    Returns:
        Population: A new population containing only the selected individuals.
    """
    new_individuals = population.individuals[indices]
    new_ages = population.age[indices]

    if population.already_eval:
        new_fitness = population.fitness[indices]
        new_penalty = population.penalty[indices]
        new_score = population.score[indices]
    else:
        new_fitness = None
        new_penalty = None
        new_score = None
    
    return Population(
        individuals=new_individuals,
        generation=population.generation,
        age=new_ages,
        fitness=new_fitness,
        penalty=new_penalty,
        score=new_score)

def cat(populations:list[Population])->Population:
    """Concatenate multiple populations into a single population.

    Args:
        populations (list[Population]): A list of populations to concatenate.
    Returns:
        Population: A new population containing all individuals from the input populations.
    """
    for population in populations:
        assert population.already_eval == populations[0].already_eval, "All populations must have the same evaluation status."
    all_individuals = torch.cat([pop.individuals for pop in populations], dim=0)
    all_ages = torch.cat([pop.age for pop in populations], dim=0)
    generation = max(pop.generation for pop in populations)

    if all(pop.already_eval for pop in populations):
        all_fitness = torch.cat([pop.fitness for pop in populations], dim=0)
        all_penalty = torch.cat([pop.penalty for pop in populations], dim=0)
        all_score = torch.cat([pop.score for pop in populations], dim=0)
    else:
        all_fitness = None
        all_penalty = None
        all_score = None
    
    return Population(
        individuals=all_individuals,
        generation=generation,
        age=all_ages,
        fitness=all_fitness,
        penalty=all_penalty,
        score=all_score)