import torch
import numpy as np

from ectorch.core.crossover import Crossover


class OnePointCrossover(Crossover):
    """One-point crossover for integer strings.

    Attributes:
        var_type (str): For OnePointCrossover, it is "Z" (integer).
        batchable (bool): True, as OnePointCrossover can be applied in batch mode.
        pair_num (int): 2, as OnePointCrossover requires two parents to produce two offspring.
        xl (torch.Tensor): Lower bounds for integer variables.
        xu (torch.Tensor): Upper bounds for integer variables.
    """
    def __init__(self, xl:torch.Tensor = None, xu:torch.Tensor = None)->None:
        super().__init__(var_type = "Z", batchable = True, pair_num = 2, xl = xl, xu = xu)
    
    def crossover(self, *parent_solutions:torch.Tensor)->tuple[torch.Tensor]:
        """Perform one-point crossover between two parent integer strings.
        
        Args:
            parent_solutions (torch.Tensor): Two parent integer strings of shape (var_dim,).
        Returns:
            tuple[torch.Tensor]: The two offspring integer strings.
        """
        parent1, parent2 = parent_solutions
        var_dim = parent1.shape[0]
        
        assert var_dim >= 2, "Variable dimension must be greater than 2 for one-point crossover."
        point = np.random.randint(1, var_dim)
        
        offspring1 = torch.cat((parent1[:point], parent2[point:]), dim=0)
        offspring2 = torch.cat((parent2[:point], parent1[point:]), dim=0)
        
        return offspring1, offspring2