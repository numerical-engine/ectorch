import torch
from ectorch.core.population import Population
from ectorch import utils

class Crossover_core:
    def __init__(self, offspring_num:int = 2, parent_num:int = 2)->None:
        self.offspring_num = offspring_num
        self.parent_num = parent_num
    
    def to(device:str)->None:
        pass

    def __call__(self, parents:Population, xl:torch.Tensor = None, xu:torch.Tensor = None)->Population:
        assert len(parents) % self.parent_num == 0, "The number of parents must be divisible by parent_num"
        indices = torch.randperm(len(parents), dtype = torch.int64, device=parents.device).view(-1, self.parent_num)
        individuals_list = [utils.population.squeeze(parents, indices[i]).individuals for i in range(indices.size(0))]
        offspring_individuals = torch.cat([self.crossover(individuals) for individuals in individuals_list])

        if xl is None: xl = -float("inf")
        if xu is None: xu = float("inf")
        offspring_individuals = torch.clamp(offspring_individuals, min=xl, max=xu)
        return Population(
            individuals=offspring_individuals,
            generation=parents.generation,
            age=torch.zeros(offspring_individuals.size(0), device=parents.device)
        )
    
    def crossover(self, parents_individuals:torch.Tensor)->torch.Tensor:
        """Perform crossover on the given parent individuals.

        Args:
            parents_individuals (torch.Tensor): The parent individuals to crossover. Shape is (parent_num, num_parameters).

        Raises:
            NotImplementedError: If the crossover method is not implemented.

        Returns:
            torch.Tensor: The offspring individuals resulting from the crossover. The shape should be (offspring_num, num_parameters).
        """
        raise NotImplementedError("Crossover method must be implemented in subclasses")