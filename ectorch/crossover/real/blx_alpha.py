import torch
import numpy as np

from ectorch.core.crossover import Crossover

class BLXAlphaCrossover(Crossover):
    """BLX-alpha crossover for real-valued vectors.

    Attributes:
        alpha (float): The alpha parameter for BLX-alpha crossover.
        var_type (str): For BLXAlphaCrossover, it is "R" (real).
        batchable (bool): True, as BLXAlphaCrossover can be applied in batch mode.
        pair_num (int): 2, as BLXAlphaCrossover requires two parents to produce two offspring.
        xl (torch.Tensor): The lower bound for the crossover operation. If None, no lower bound is applied.
        xu (torch.Tensor): The upper bound for the crossover operation. If None, no upper bound is applied.
    """
    def __init__(self, alpha: float = 0.5, xl: torch.Tensor = None, xu: torch.Tensor = None) -> None:
        super().__init__(var_type="R", batchable=True, pair_num=2, xl=xl, xu=xu)
        assert 0.0 < alpha < 1.0, "Alpha parameter must be in (0, 1)."
        self.alpha = alpha
    
    def crossover(self, *parent_solutions: torch.Tensor) -> tuple[torch.Tensor]:
        """Perform BLX-alpha crossover between two parent real-valued vectors.

        Args:
            parent_solutions (torch.Tensor): Two parent real-valued vectors of shape (var_dim,).
        Returns:
            tuple[torch.Tensor]: The two offspring real-valued vectors.
        """
        parent1, parent2 = parent_solutions

        c_min = torch.min(parent1, parent2)
        c_max = torch.max(parent1, parent2)
        length = c_max - c_min

        lower_bound = c_min - self.alpha * length
        upper_bound = c_max + self.alpha * length

        offspring1 = lower_bound + (upper_bound - lower_bound) * torch.rand_like(lower_bound)
        offspring2 = lower_bound + (upper_bound - lower_bound) * torch.rand_like(lower_bound)

        return offspring1, offspring2