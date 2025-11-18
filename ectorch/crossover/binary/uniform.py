import torch
import numpy as np

from ectorch.core.crossover import Crossover

class UniformCrossover(Crossover):
    """Uniform crossover for binary strings.
    
    Attributes:
        p (float): Crossover probability.
        var_type (str): For OnePointCrossover, it is "B" (binary).
        batchable (bool): True, as OnePointCrossover can be applied in batch mode.
        pair_num (int): 2, as OnePointCrossover requires two parents to produce two offspring.
        xl (torch.Tensor): None, as there are no bounds for binary variables.
        xu (torch.Tensor): None, as there are no bounds for binary variables.
    """
    def __init__(self, p:float = 0.5)->None:
        super().__init__(var_type = "B", batchable = True, pair_num = 2, xl = None, xu = None)
        assert 0.0 < p < 1.0, "Crossover probability p must be in [0, 1]."
        self.p = p
    
    def crossover(self, *parent_solutions):
        """Perform uniform crossover between two parent binary strings.

        Args:
            parent_solutions (torch.Tensor): Two parent binary strings of shape (var_dim,).
        Returns:
            tuple[torch.Tensor]: The two offspring binary strings.
        """
        parent1, parent2 = parent_solutions
        var_dim = parent1.shape[0]

        mask = torch.rand(var_dim, device=parent1.device) < self.p
        offspring1 = torch.where(mask, parent1, parent2)
        offspring2 = torch.where(mask, parent2, parent1)
        return offspring1, offspring2