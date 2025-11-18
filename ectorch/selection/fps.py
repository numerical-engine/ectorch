import torch

from ectorch.core.selection import Selection

class FPSelection(Selection):
    """Fitness Proportional Selection (FPS) operator for selecting diverse individuals from a population.
    
    Attributes:
        window (torch.Tensor): Optional tensor defining the selection window.
    """
    def __init__(self, windowing:torch.Tensor = None)->None:
        self.windowing = 0.0 if windowing is None else windowing
    
    def to(self, device:str)->None:
        """Moves the windowing tensor to the specified device.

        Args:
            device (str): The target device to move the windowing tensor to.
        """
        if self.windowing is not None:
            self.windowing = self.windowing.to(device)

    def select(self, scores:torch.Tensor, selection_num:int)->torch.Tensor:
        """Select individuals based on fitness proportional selection.

        Args:
            scores (torch.Tensor): The scores of the individuals in the population. The shape is (population_size,).
            selection_num (int): The number of individuals to select.
        Returns:
            torch.Tensor: The indices of the selected individuals. The shape is (selection_num,).
                - data type is torch.long
        """
        min_score = torch.min(scores)
        adjusted_scores = scores - min_score + self.windowing
        
        total_score = torch.sum(adjusted_scores)
        if total_score.item() == 0.0:
            probabilities = torch.ones_like(adjusted_scores) / adjusted_scores.shape[0]
        else:
            probabilities = adjusted_scores / total_score
        
        selected_indices = torch.multinomial(probabilities, selection_num, replacement=True)
        return selected_indices