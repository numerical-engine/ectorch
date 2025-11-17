import torch

from ectorch.core.mutation import Mutation

class CreepMutation(Mutation):
    """Mutation that creeps an integer gene by a small random value with probability within specified bounds.
    
    Attributes:
        p (float): The probability of mutating each integer gene.
        step_size (int): The maximum step size for creeping.
        var_type (str): The type of variable, set to "Z" for integer variables.
        batchable (bool): Indicates that this mutation can be applied in batch mode.
        xl (torch.Tensor): The lower bounds for the integer variables.
        xu (torch.Tensor): The upper bounds for the integer variables.
    """
    def __init__(self, p:float, xl:torch.Tensor, xu:torch.Tensor, step_size:int=1)->None:
        super().__init__(var_type="Z", batchable=True, xl=xl, xu=xu)
        self.p = p
        self.step_size = step_size
    
    def mutate(self, solution:torch.Tensor)->torch.Tensor:
        """Mutate the given solution by creeping its integer genes by a small random value.

        Args:
            solution (torch.Tensor): The input tensor to mutate. The shape is (variable_dimension, ).
        Returns:
            torch.Tensor: The mutated tensor. The shape is (variable_dimension, ).
        """
        creep_vals = torch.cat([torch.randint(low=-self.step_size, high=self.step_size + 1, size=(1,), device=solution.device) for _ in range(solution.shape[0])])
        mutation_mask = torch.rand(solution.shape, device=solution.device) < self.p
        mutated_solution = torch.where(mutation_mask, solution + creep_vals, solution)
        return mutated_solution