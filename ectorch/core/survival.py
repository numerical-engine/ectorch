import torch

from ectorch import config
from ectorch.utils import population_utils

class Survival:
    """Abstract base class for survival selection operators in evolutionary algorithms.
    """
    def __call__(self, parents:"Population", offspring:"Population", environment:"Environment", reset_score:bool = False)->"Population":
        """Return population of next generation.

        Args:
            parents (Population): parents population
            offspring (Population): offspring population
            environment (Environment): environment instance
            reset_score (bool, optional): whether to reset scores. Defaults to False.
        Returns:
            Population: next generation population
        """
        if reset_score:
            parents.reset()
            offspring.reset()
        
        if not parents.already_scored or not offspring.already_scored:
            mu = len(parents); lam = len(offspring)
            population = population_utils.cat([parents, offspring])
            environment.run(population)
            parents = population.index_select(torch.arange(0, mu, device=population.device))
            offspring = population.index_select(torch.arange(mu, mu + lam, device=population.device))
        
        population = self.survive(parents, offspring)
    
        return population
    
    def to(self, device:str)->None:
        """Moves any internal tensors to the specified device.

        Args:
            device (str): The target device to move internal tensors to.
        """
        pass

    def survive(self, parents:"Population", offspring:"Population")->"Population":
        """Select individuals from parents and offspring to form the next generation.

        This method should be implemented by subclasses.

        Args:
            parents (Population): The parents population.
            offspring (Population): The offspring population.
        Returns:
            Population: The next generation population.
        """
        raise NotImplementedError("Subclasses must implement this method.")