import torch
import torch.nn as nn

from ectorch.core.population import Population
from ectorch.core.function import Function, PenaltyLower, PenaltyUpper, PenaltyFunction

class Environment_core:
    def __init__(self, obj_functions:list[Function], xl:torch.Tensor = None, xu:torch.Tensor = None, penalty_functions:list[PenaltyFunction] = [])->None:
        self.obj_functions = obj_functions
        self.penalty_functions = penalty_functions

        if xl is not None:
            self.penalty_functions.append(PenaltyLower(xl))
        if xu is not None:
            self.penalty_functions.append(PenaltyUpper(xu))

    def set_fitness(self, population:Population)->None:
        """Set the fitness of the population.

        Args:
            population (Population): The population to set the fitness for.
        """
        if not population.already_fit:
            population.fitness = torch.stack([objf(population) for objf in self.obj_functions], dim=1)

            if self.penalty_functions:
                population.penalty = torch.stack([penf(population) for penf in self.penalty_functions], dim=1)
            else:
                population.penalty = torch.zeros((population.size, 1), device=population.device)

    def set_score(self, population:Population)->None:
        """Set the score of the population.

        Args:
            population (Population): The population to set the score for.
        """
        if not population.already_eval:
            raise NotImplementedError("set_score method must be implemented in subclasses.")
        else:
            pass


    def evaluate(self, population:Population)->None:
        """Set the fitness, penalty and score of the population.

        Args:
            population (Population): The population to evaluate.
        """
        self.set_fitness(population)
        self.set_score(population)