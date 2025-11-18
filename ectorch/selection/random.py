import torch

from ectorch.core.selection import Selection

class RandomSelection(Selection):
    """Randomly select individuals from the population.

    Note:
        - Replacement is allowed.
    """
    def select(self, scores:torch.Tensor, selection_num:int)->torch.Tensor:
        population_size = scores.shape[0]
        indices = torch.randint(0, population_size, (selection_num,), device=scores.device)
        return indices