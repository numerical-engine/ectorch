import torch

from ectorch.core.population import Population
from ectorch import utils

class Selection_core:
    def __call__(self, population: Population, selection_num:int) -> Population:
        assert population.already_eval, "Population must be evaluated before selection."
        indices = self.select(population.score, selection_num)

        return utils.population.squeeze(population, indices)
    
    def to(self, device:str)->None:
        pass
    
    def select(self, scores: torch.Tensor, selection_num:int) -> torch.Tensor:
        """Select individuals based on their scores.

        Args:
            scores (torch.Tensor): The scores of the individuals. Shape is (N,), where N is the number of individuals.
            selection_num (int): The number of individuals to select.
        Returns:
            torch.Tensor: The indices of the selected individuals. Shape is (selection_num,).
        """
        raise NotImplementedError