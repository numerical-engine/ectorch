import torch

from ectorch import config
import sys

class Crossover:
    """Abstract base class for crossover operators.
    
    Attributes:
        var_type (str): The type of variable the crossover operator is designed for.
            - var_type should be one of config.var_keys.
        batchable (bool): Whether the crossover operator can be applied in batch mode.
        pair_num (int): The number of parents required for the crossover operation.
        xl (torch.Tensor): The lower bound for the crossover operation. If None, no lower bound is applied.
        xu (torch.Tensor): The upper bound for the crossover operation. If None, no upper bound is applied.
    
    Note:
        - In ectorch, The number of parents required for crossover is always equal to the number of offspring produced, because of simplicity.
    """
    def __init__(self, var_type:str, batchable:bool, pair_num:int = 2, xl:torch.Tensor = None, xu:torch.Tensor = None)->None:
        assert var_type in config.var_keys, f"Invalid variable type: {var_type}."
        self.var_type = var_type
        self.batchable = batchable
        self.pair_num = pair_num
        self.xl = -torch.inf if xl is None else xl
        self.xu = torch.inf if xu is None else xu
    
    def to(self, device:torch.device)->None:
        """Moves the crossover operator to the specified device.

        Args:
            device (torch.device): The device to move the crossover operator to.
        """
        self.xl = self.xl.to(device)
        self.xu = self.xu.to(device)
    
    def create_pairs(self, solution:torch.Tensor)->tuple[torch.Tensor]:
        """Creates pairs of solutions for crossover.

        Args:
            solution (torch.Tensor): The input solutions of shape (population_size, var_dim).
        Returns:
            tuple[torch.Tensor]: A tuple containing the paired solutions.
                - The length of the tuple is self.pair_num, and each element has shape (population_size % self.pair_num, var_dim).
        """
        assert solution.shape[0] % self.pair_num == 0, "Population size must be divisible by pair_num."
        shuffled_indices = torch.randperm(solution.shape[0], device=solution.device)
        split_solutions = torch.chunk(solution[shuffled_indices], self.pair_num, dim=0)
        return split_solutions


    def __call__(self, parents:"Population")->"Population":
        """Applies the crossover operator to a population of parents.

        Args:
            parents (Population): The population of parents.
        Returns:
            Population: The population of offspring.
        """
        solution = parents.variables[self.var_type] #torch.Tensor of shape (parent_num, population_size, var_dim)
        solutions = self.create_pairs(solution)

        if self.batchable:
            batch_crossover = torch.vmap(self.crossover, randomness="different")
            offspring_solutions = torch.cat(batch_crossover(*solutions))
        else:
            offspring_solutions = torch.cat([torch.stack(self.crossover(*parent_solutions)) for parent_solutions in zip(*solutions)])
        
        offspring_solutions = torch.clamp(offspring_solutions, min=self.xl, max=self.xu)
        new_variables = parents.variables.copy()
        new_variables[self.var_type] = offspring_solutions
        
        offspring_population = type(parents)(
            variables=new_variables,
            age=parents.age,
            generation=parents.generation,
            fitness = None,
            penalty = None,
            score = None,)
        
        return offspring_population
    
    def crossover(self, *parent_solutions:torch.Tensor)->tuple[torch.Tensor]:
        """Performs crossover between parent solutions.

        Args:
            parent_solutions (torch.Tensor): The parent solutions to crossover. 
                - The shape is (variable_dimension, ).
                - The number of parent_solutions is self.parent_num.
        Returns:
            tuple[torch.Tensor]: The offspring solutions generated from the parents.
                - Each torch.Tensor has shape (variable_dimension, ).
                - The length of the tuple is self.offspring_num.
        """
        raise NotImplementedError("Crossover method must be implemented in subclasses.")