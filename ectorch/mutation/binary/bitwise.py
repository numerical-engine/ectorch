import torch

from ectorch.core.mutation import Mutation

class BitwiseMutation(Mutation):
    """Bitwise mutation operator for binary variables.
    
    Attributes:
        p (float): The probability of flipping each bit.
    """
    def __init__(self, p:float)->None:
        super().__init__(var_type="B", batchable=True)
        self.p = p
    def mutate(self, solution:torch.Tensor)->torch.Tensor:
        """Applies bitwise mutation to one binary solution.

        Args:
            solution (torch.Tensor): The binary solution to mutate. The shape is (variable_dimension, ).
        Returns:
            torch.Tensor: The mutated binary solution. The shape is (variable_dimension, ).
        Note:
            The data type of solution is config.dtype (may be torch.float32).
        """
        flip_mask = torch.rand_like(solution) < self.p
        mutated_solution = torch.where(flip_mask, 1 - solution, solution)
        return mutated_solution