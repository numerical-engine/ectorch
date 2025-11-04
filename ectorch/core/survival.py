import torch
from ectorch.core.population import Population
from ectorch import utils

class Survival_core:
    def __call__(self, parents:Population, offsprings:Population, environment:"Environment" = None, reset_score:bool = True)->Population:
        if (parents.already_eval == offsprings.already_eval) or reset_score:
            parents.reset()
            offsprings.reset()
        population = utils.cat([parents, offsprings])

        if not population.already_eval:
            environment.evaluate(population)
        parents = utils.squeeze(population, torch.arange(0, len(parents), dtype = torch.int64, device=population.device))
        offsprings = utils.squeeze(population, torch.arange(len(parents), len(population), dtype = torch.int64, device=population.device))

        survived_population = self.survive(parents, offsprings)
        assert len(survived_population) == len(parents), "The number of survived individuals must be equal to the number of parents."
        return survived_population

    def to(device:str)->None:
        pass

    def survive(self, parents:Population, offsprings:Population)->Population:
        """Select individuals to survive from the combined population of parents and offsprings.

        Args:
            parents (Population): The parent population.
            offsprings (Population): The offspring population.
        Returns:
            Population: The surviving individuals from the combined population.
        """
        raise NotImplementedError("Survive method must be implemented in subclasses.")