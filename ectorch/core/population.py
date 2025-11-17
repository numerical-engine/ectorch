import torch

from ectorch.config import config
from ectorch.utils import population_utils

class Population:
    """The class managing population of evolutionary computing

    Attributes:
        generation (int): The current generation number of the population.
        age (torch.Tensor): A tensor of shape (population_size,) representing the age of each individual in the population.
        variables (dict[str, torch.Tensor]): A dictionary representing the variables of the population.
            - Each key is the variable type which should be one of config.var_keys.
            - Each value is a tensor of shape (population_size, variable_dimension) representing the values of the variable.
        fitness (torch.Tensor): A tensor of shape (population_size, num_objectives) representing the objective function values for each individual.
            - Smaller values are considered better.
            - If None, it indicates that the objective functions has not been evaluated yet.
        penalty (torch.Tensor): A tensor of shape (population_size, num_penalties) representing the penalty values for each individual.
            - If None, it indicates that no penalties are applied or evaluated yet.
            - If all penalty values are zero, it indicates this individual is in a feasible region.
        score (torch.Tensor): A tensor of shape (population_size,) representing the overall score for each individual.
            - Higher values are considered better.
            - If None, it indicates that the score has not been calculated yet.
    """
    def __init__(self,
                variables:dict[str, torch.Tensor],
                age:torch.Tensor = None,
                generation:int = 0,
                fitness:torch.Tensor = None,
                penalty:torch.Tensor = None,
                score:torch.Tensor = None,
                )->None:
        assert population_utils.check_variables(variables), "Invalid variables provided."
        self.variables = variables

        self.generation = generation

        num_population = next(iter(variables.values())).size(0)
        self.age = age if age is not None else torch.zeros(num_population, dtype=torch.int64)
        
        self.fitness = fitness
        self.penalty = penalty
        self.score = score
    
    def __len__(self)->int:
        """Returns the population size.

        Returns:
            int: The number of individuals in the population.
        """
        return self.age.size(0)
    
    @property
    def device(self)->str:
        """Returns the device of the population variables.

        Returns:
            str: The device where the population variables are stored.
        """
        return self.age.device
    
    def to(self, device:str)->None:
        """Moves all tensors in the population to the specified device.

        Args:
            device (str): The target device to move the population variables to.
        """
        for var_type in self.variables:
            self.variables[var_type] = self.variables[var_type].to(device)
        self.age = self.age.to(device)
        if self.fitness is not None:
            self.fitness = self.fitness.to(device)
        if self.penalty is not None:
            self.penalty = self.penalty.to(device)
        if self.score is not None:
            self.score = self.score.to(device)

    @property
    def already_evaluated(self)->bool:
        """Checks if the population has been evaluated.

        If the fitness and penalty attributes are not None, it indicates that the population has been evaluated.
        Returns:
            bool: True if the population has been evaluated, False otherwise.
        """
        return (self.fitness is not None) and (self.penalty is not None)
    
    @property
    def already_scored(self)->bool:
        """Checks if the population has been scored.

        If the score attribute is not None, it indicates that the population has been scored.
        Returns:
            bool: True if the population has been scored, False otherwise.
        """
        return (self.score is not None) and self.already_evaluated
    
    def clone(self)->'Population':
        """Creates a clone of the current population.

        Returns:
            Population: A new Population object with the same attributes as the current one.
        """
        return Population(
            variables={var_type: {"type": var_type, "value": value.clone()} for var_type, value in self.variables.items()},
            age=self.age.clone(),
            generation=self.generation,
            fitness=self.fitness.clone() if self.fitness is not None else None,
            penalty=self.penalty.clone() if self.penalty is not None else None,
            score=self.score.clone() if self.score is not None else None,
        )