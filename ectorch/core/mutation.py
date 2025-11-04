import torch
from ectorch.core.population import Population


class Mutation_core:
    def __call__(self, population:Population, xl:torch.Tensor = None, xu:torch.Tensor = None)->Population:
        mutated_individuals = torch.stack([self.mutate(individual) for individual in population.individuals])

        if xl is None: xl = -float("inf")
        if xu is None: xu = float("inf")
        mutated_individuals = torch.clamp(mutated_individuals, min=xl, max=xu)
        return Population(
            individuals=mutated_individuals,
            generation=population.generation,
            age=population.age.clone()
        )
    
    def to(device:str)->None:
        pass
    
    def mutate(self, individual:torch.Tensor)->torch.Tensor:
        """Perform mutation on the given individual.

        Args:
            individual (torch.Tensor): The individual to mutate. Shape is (num_parameters,).
        Raises:
            NotImplementedError: If the mutation method is not implemented.
        Returns:
            torch.Tensor: The mutated individual. Shape is (num_parameters,).
        """
        raise NotImplementedError("Mutation method must be implemented in subclasses")