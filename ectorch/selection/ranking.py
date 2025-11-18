import torch

from ectorch.core.selection import Selection

class RankingSelection(Selection):
    """Ranking Selection operator for selecting individuals based on their rank in the population.
    
    Attributes:
        s (float): Selection pressure parameter, typically in the range [1.0, 2.0].
    
    """
    def __init__(self, s:float = 1.5)->None:
        self.s = s

    def select(self, scores:torch.Tensor, selection_num:int)->torch.Tensor:
        """Select individuals based on their ranking.

        Args:
            scores (torch.Tensor): The scores of the individuals in the population. The shape is (population_size,).
            selection_num (int): The number of individuals to select.
        Returns:
            torch.Tensor: The indices of the selected individuals. The shape is (selection_num,).
                - data type is torch.long
        """
        _, sorted_indices = torch.sort(scores)
        ranks = torch.zeros_like(sorted_indices)
        ranks[sorted_indices] = torch.arange(len(scores), device=scores.device)
        
        N = len(scores)
        selection_probabilities = (2 - self.s) / N + (2 * ranks * (self.s - 1)) / (N * (N - 1))
        selected_indices = torch.multinomial(selection_probabilities, selection_num, replacement=True)
        return selected_indices