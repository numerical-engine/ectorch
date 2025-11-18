import torch
import numpy as np

from ectorch.core.crossover import Crossover

class UniformCrossover(Crossover):
    """Uniform crossover for real-valued vectors.

    Attributes:
        p (float): Crossover probability.
        var_type (str): For UniformCrossover, it is "R" (real).
        batchable (bool): True, as UniformCrossover can be applied in batch mode.
        pair_num (int): 2, as UniformCrossover requires two parents to produce two offspring.
        xl (torch.Tensor): The lower bound for the crossover operation. If None, no lower bound is applied.
        xu (torch.Tensor): The upper bound for the crossover operation. If None, no upper bound is applied.
    """
    def __init__(self, p:float = 0.5, xl:torch.Tensor = None, xu:torch.Tensor = None)->None:
        super().__init__(var_type = "R", batchable = True, pair_num = 2, xl = xl, xu = xu)
        assert 0.0 < p < 1.0, "Crossover probability p must be in [0, 1]."
        self.p = p
    
    def crossover(self, *parent_solutions:torch.Tensor)->tuple[torch.Tensor]:
        """Perform uniform crossover between two parent real-valued vectors.

        Args:
            parent_solutions (torch.Tensor): Two parent real-valued vectors of shape (var_dim,).
        Returns:
            tuple[torch.Tensor]: The two offspring real-valued vectors.
        """
        parent1, parent2 = parent_solutions
        var_dim = parent1.shape[0]

        mask = torch.rand(var_dim, device=parent1.device) < self.p
        offspring1 = torch.where(mask, parent1, parent2)
        offspring2 = torch.where(mask, parent2, parent1)

        return offspring1, offspring2