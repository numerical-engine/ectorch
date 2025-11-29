import torch

class Optimizer:
    def __call__(self, population:"Population", environment:"Environment")->"Population":
        """Update population

        Args:
            population (Population): The population of present generation.
            environment (Environment): The environment used for optimization.
        Returns:
            Population: The next generation population.
        """
        if not population.aleady_scored:
            environment.run(population)
        next_population = population.clone()
        next_population.generation += 1
        next_population.age += 1
        next_population = self.run(next_population, environment)

        return next_population
    
    def run(self, population:"Population", environment:"Environment")->"Population":
        """Runs the optimization process to generate the next population.

        This method should be implemented by subclasses to define specific optimization strategies.

        Args:
            population (Population): The current population.
            environment (Environment): The environment used for optimization.
        Returns:
            Population: The next generation population.
        """
        raise NotImplementedError("Subclasses must implement this method.")