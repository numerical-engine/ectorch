import torch

from ectorch import config

class Mutation:
    """Abstract base class for mutation operators in evolutionary computing.

    Attributes:
        var_type (str, optional): The type of variable the mutation operator is designed for.
            - var_type should be one of config.var_keys.
        xl (torch.Tensor): The lower bounds for the variables. If None, no lower bound is applied.
        xu (torch.Tensor): The upper bounds for the variables. If None, no upper bound is applied.
        batchable (bool): Whether the mutation operator can be applied in batch mode.
    """
    def __init__(self, var_type:str, batchable:bool, xl:torch.Tensor = None, xu:torch.Tensor = None)->None:
        assert var_type in config.var_keys, f"Invalid variable type: {var_type}."
        self.var_type = var_type
        self.batchable = batchable
        self.xl = -torch.inf if xl is None else xl
        self.xu = torch.inf if xu is None else xu
    
    def __call__(self, population:"Population")->"Population":
        """Applies the mutation operator to a population.

        Args:
            population (Population): The population to mutate.
        Returns:
            Population: The mutated population.
        """
        solution = population.variables[self.var_type]

        if self.batchable:
            batch_mutate = torch.vmap(self.mutate, randomness="different")
            new_solution = batch_mutate(solution)
        else:
            new_solution = torch.stack([self.mutate(sol) for sol in solution], dim=0)
        
        new_solution = torch.clamp(new_solution, min=self.xl, max=self.xu)
        new_variables = population.variables.copy()
        new_variables[self.var_type] = new_solution
        new_population = type(population)(
            variables=new_variables,
            age=population.age,
            generation=population.generation,
            fitness = None,
            penalty = None,
            score = None,)
        
        return new_population

    def to(self, device:torch.device)->None:
        """Moves the mutation operator to the specified device.

        Args:
            device (torch.device): The device to move the mutation operator to.
        """
        self.xl = self.xl.to(device)
        self.xu = self.xu.to(device)

    def mutate(self, solution:torch.Tensor)->torch.Tensor:
        """Applies the mutation operator to one solution.

        Args:
            solution (torch.Tensor): The solution to mutate. The shape is (variable_dimension, ).
        Returns:
            torch.Tensor: The mutated solution. The shape is (variable_dimension, ).
        """
        raise NotImplementedError("Mutation operator must implement the mutate method.")