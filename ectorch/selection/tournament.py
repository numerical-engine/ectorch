import torch
import sys

from ectorch.core.selection import Selection

class TournamentSelection(Selection):
    """Tournament Selection operator for selecting individuals through tournaments.

    Attributes:
        tournament_size (int): The number of individuals competing in each tournament.
    """
    def __init__(self, tournament_size:int = 3)->None:
        self.tournament_size = tournament_size
    
    def select(self, scores:torch.Tensor, selection_num:int)->torch.Tensor:
        """Select individuals based on tournament selection.

        Args:
            scores (torch.Tensor): The scores of the individuals in the population. The shape is (population_size,).
            selection_num (int): The number of individuals to select.
        Returns:
            torch.Tensor: The indices of the selected individuals. The shape is (selection_num,).
                - data type is torch.long
        """
        tournament_indices = torch.randint(0, len(scores), (selection_num, self.tournament_size), device=scores.device)
        tournament_scores = scores[tournament_indices]
        winner_indices_in_tournament = torch.argmax(tournament_scores, dim=1)
        selected_indices = tournament_indices[torch.arange(selection_num, device=scores.device), winner_indices_in_tournament]
        return selected_indices