import torch

from ectorch.core.mutation import Mutation

class ResetMutation(Mutation):
    """Mutation that resets an integer gene to a new random value within specified bounds.

    This mutation for the cardinal attributes (e.g., {"North", "East", "South", "West"}).

    Attributes:
        p (float): The probability of mutating each integer gene.
        var_type (str): The type of variable, set to "Z" for integer variables.
        batchable (bool): Indicates that this mutation can be applied in batch mode.
        xl (torch.Tensor): The lower bounds for the integer variables.
        xu (torch.Tensor): The upper bounds for the integer variables.
    """
    def __init__(self, p:float, xl:torch.Tensor, xu:torch.Tensor)->None:
        super().__init__(var_type="Z", batchable=True, xl=xl, xu=xu)
        self.p = p

    def mutate(self, solution:torch.Tensor)->torch.Tensor:
        """Mutate the given solution by resetting its integer genes to new random values.

        Args:
            solution (torch.Tensor): The input tensor to mutate. The shape is (variable_dimension, ).
        Returns:
            torch.Tensor: The mutated tensor. The shape is (variable_dimension, ).
        """
        rand_vals = torch.cat([torch.randint(low=int(self.xl[i].item()), high=int(self.xu[i].item()) + 1, size=(1,), device=solution.device) for i in range(solution.shape[0])])
        mutation_mask = torch.rand(solution.shape, device=solution.device) < self.p
        mutated_solution = torch.where(mutation_mask, rand_vals, solution)
        return mutated_solution