import torch

from ectorch.core.mutation import Mutation

class NormalMutation(Mutation):
    """Normal mutation operator for real-valued variables.
    
    Attributes:
        sigma (torch.Tensor): The standard deviation of the normal distribution. The shape is (variable_dimension, ).
    """
    def __init__(self, sigma:torch.Tensor, xl:torch.Tensor, xu:torch.Tensor)->None:
        super().__init__(var_type="R", batchable=True, xl=xl, xu=xu)
        self.sigma = sigma

    def mutate(self, solution:torch.Tensor)->torch.Tensor:
        """Applies normal mutation to one real-valued solution.

        Args:
            solution (torch.Tensor): The real-valued solution to mutate. The shape is (variable_dimension, ).
        Returns:
            torch.Tensor: The mutated real-valued solution. The shape is (variable_dimension, ).
        """
        noise = torch.randn_like(solution) * self.sigma
        mutated_solution = solution + noise
        return mutated_solution