import torch
import numpy as np

from ectorch.core.crossover import Crossover

class SBXCrossover(Crossover):
    """Simulated Binary Crossover (SBX) for real-valued vectors.

    Attributes:
        p (float): The probability parameter for SBX.
        eta (float): The distribution index for SBX.
        var_type (str): For SBXCrossover, it is "R" (real).
        batchable (bool): True, as SBXCrossover can be applied in batch mode.
        pair_num (int): 2, as SBXCrossover requires two parents to produce two offspring.
        xl (torch.Tensor): The lower bound for the crossover operation. If None, no lower bound is applied.
        xu (torch.Tensor): The upper bound for the crossover operation. If None, no upper bound is applied.
    """
    def __init__(self, p: float = 0.5, eta: float = 15.0, xl: torch.Tensor = None, xu: torch.Tensor = None) -> None:
        super().__init__(var_type="R", batchable=True, pair_num=2, xl=xl, xu=xu)
        assert eta > 0.0, "Distribution index eta must be positive."
        assert 0.0 < p <= 1.0, "Probability p must be in (0, 1]."
        self.p = p
        self.eta = eta
    
    def crossover(self, *parent_solutions: torch.Tensor) -> tuple[torch.Tensor]:
        """Perform SBX crossover between two parent real-valued vectors.

        Args:
            parent_solutions (torch.Tensor): Two parent real-valued vectors of shape (var_dim,).
        Returns:
            tuple[torch.Tensor]: The two offspring real-valued vectors.
        """
        parent1, parent2 = parent_solutions
        rand = torch.rand_like(parent1)

        beta = torch.where(rand <= 0.5, (2.0 * rand) ** (1.0 / (self.eta + 1.0)), (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / (self.eta + 1.0)))

        mask = torch.rand_like(parent1) <= self.p

        offspring1 = torch.where(mask, 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2), parent1)
        offspring2 = torch.where(mask, 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2), parent2)

        return offspring1, offspring2