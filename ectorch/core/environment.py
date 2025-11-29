import torch

from ectorch import config

class Environment:
    """Abstract class for environments in evolutionary algorithms.
    
    Attributes:
        functions (list[Function]): A list of objective function instances.
        penalty_functions (list[PenaltyFunction]): A list of penalty function instances. If None, no penalty functions are applied.
    """
    def __init__(self, functions:list["Function"], penalty_functions:list["PenaltyFunction"] = None)->None:
        self.functions = functions
        self.penalty_functions = penalty_functions
    
    def get_score(self, population:"Population")->torch.Tensor:
        """Calculates the overall score for the given population.

        Args:
            population (Population): The population to be scored.
        Returns:
            torch.Tensor: The score tensor of shape (population_size,).
        """
        raise NotImplementedError("The get_score method should be implemented in subclasses.")
    
    def evaluate(self, population:"Population")->tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the fitness and penalties for the given population.

        Args:
            population (Population): The population to be evaluated.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - fitness (torch.Tensor): The fitness tensor of shape (population_size, num_objectives).
                - penalty (torch.Tensor): The penalty tensor of shape (population_size, num_penalties).
        """
        fitness = torch.stack([func(population) for func in self.functions], dim=1)
        penalty = torch.stack([pfunc(population) for pfunc in self.penalty_functions], dim=1) if self.penalty_functions is not None else None
        return fitness, penalty

    def run(self, population:"Population")->None:
        """Set fitness, penalties and scores for the given population.

        Args:
            population (Population): The population to be evaluated.
        """
        if population.already_scored:
            pass
        else:
            if not(population.already_evaluated):
                population.fitness, population.penalty = self.evaluate(population)
            population.score = self.get_score(population)
