import torch

from ectorch import Survival
from ectorch.utils import population_utils

class mu_to_lambda(Survival):
    def survive(self, parents:"Population", offspring:"Population")->"Population":
        """Select the best 'lambda' individuals from the offspring population.

        Args:
            parents (Population): The parents population.
            offspring (Population): The offspring population.
        Returns:
            Population: The next generation population consisting of the best 'mu' individuals from the offspring.
        
        Note:
            - Since ectorch assume the number of parents is equal to the number of offspring, this method simply returns the offspring population.
        """
        return offspring

class mu_plus_lambda(Survival):
    def survive(self, parents:"Population", offspring:"Population")->"Population":
        """Select the best 'mu' individuals from the combined parents and offspring populations.

        Args:
            parents (Population): The parents population.
            offspring (Population): The offspring population.
        Returns:
            Population: The next generation population consisting of the best 'mu' individuals from the combined parents and offspring.
        """
        mu = len(parents)
        combined_population = population_utils.cat([parents, offspring])
        sorted_population = combined_population.sort()
        next_generation = sorted_population.index_select(torch.arange(0, mu, device=combined_population.device))
        return next_generation