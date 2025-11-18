import torch

from ectorch import config

class Selection:
    """Abstract base class for selection operators in evolutionary algorithms.
    """
    def __call__(self, population:"Population", selection_num:int)->"Population":
        """Select individuals from the population.

        Args:
            population (Population): The population to select from.
            selection_num (int): The number of individuals to select.
        """
        assert population.already_scored, "Population must be scored before selection."
        indices = self.select(population.score, selection_num)

        return population.index_select(indices)

    def to(self, device:str)->None:
        """Moves any internal tensors to the specified device.

        Args:
            device (str): The target device to move internal tensors to.
        """
        pass
    
    def select(self, scores:torch.Tensor, selection_num:int)->torch.Tensor:
        """Select individuals based on their scores.

        This method should be implemented by subclasses.

        Args:
            scores (torch.Tensor): The scores of the individuals in the population. The shape is (population_size,).
            selection_num (int): The number of individuals to select.

        Returns:
            torch.Tensor: The indices of the selected individuals. The shape is (selection_num,).
                - data type is torch.long
        """
        raise NotImplementedError("Subclasses must implement this method.")