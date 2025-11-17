import torch

from ectorch.core.mutation import Mutation

class PBMMutation(Mutation):
    """Polynomial Bounded Mutation (PBM) operator for real-valued variables.
    
    Attributes:
        xl (torch.Tensor): The lower bounds for the variables. The shape is (variable_dimension, ).
        xu (torch.Tensor): The upper bounds for the variables. The shape is (variable_dimension, ).
        p (float): The mutation probability for each variable.
        eta (float): The distribution index for the mutation.
        mutp (float): Precomputed value for mutation calculation.
    """
    def __init__(self, xl:torch.Tensor, xu:torch.Tensor, p:float = 0.1, eta:float=20.0)->None:
        super().__init__(var_type="R", batchable=True, xl=xl, xu=xu)
        self.p = p
        self.eta = eta
        self.mutp = 1./(self.eta + 1.)

    def mutate(self, solution:torch.Tensor)->torch.Tensor:
        """Applies Polynomial Bounded Mutation to one real-valued solution.

        Args:
            solution (torch.Tensor): The real-valued solution to mutate. The shape is (variable_dimension, ).
        Returns:
            torch.Tensor: The mutated real-valued solution. The shape is (variable_dimension, ).
        """
        U = torch.rand_like(solution)
        delta = torch.where(U < 0.5, (2.*U)**(self.mutp - 1.), 1. - (1./(2.*(1.-U)))**self.mutp)
        mask = (torch.rand_like(solution) < self.p)
        mutated_solution = solution + delta * (self.xu - self.xl) * mask

        return mutated_solution