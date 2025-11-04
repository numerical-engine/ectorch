import torch

from ectorch.core.population import Population
from ectorch.core.environment import Environment_core

class Optimizer:
    def __call__(self, population:Population, environment:Environment_core)->Population:
        """Update the population by one generation.

        Args:
            population (Population): The population to update.
            environment (Environment_core): The environment to use for evaluation.
        Returns:
            Population: The updated population.
        """
        if not population.already_eval:
            environment.evaluate(population)
        
        population_new = population.copy()
        population_new.generation += 1
        population_new.age += 1
        population_new = self.run(population_new, environment)
        if not population_new.already_eval:
            environment.evaluate(population_new)
        return population_new
    
    def run(self, population:Population, environment:Environment_core)->Population:
        """Run the optimization process.

        Args:
            population (Population): The population to optimize.
            environment (Environment_core): The environment to use for evaluation.

        Returns:
            Population: The updated population.
        """
        raise NotImplementedError("run method must be implemented in subclasses.")