import torch
import sys

from ectorch.core.selection import Selection
from ectorch.utils.score_utils import get_frontrank, get_crowding_distance

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


class MOTournamentSelection_deb(Selection):
    """Multi-Objective Tournament Selection operator for selecting individuals through tournaments.

    This selection method was proposed for treating constraints in multi-objective optimization by Deb et al.
    - The number of competitor should be 2.
    - If one individual is feasible and the other is not, the feasible one is selected.
    - Else if both individuals are feasible.
        - If they have different front ranks, the one with the lower front rank is selected.
        - Else the one with the higher crowding distance is selected.
    - Else if both individuals are infeasible, the one with smaller constraint violation is selected.
    """
    def __call__(self, population:"Population", selection_num:int)->"Population":
        """Select individuals from the population.

        Args:
            population (Population): The population to select from.
            selection_num (int): The number of individuals to select.
        """
        assert population.already_evaluated, "Population must be evaluated before selection."
        assert population.already_penalized, "Population must be penalized before selection."

        front_rank = get_frontrank(population.fitness)
        crowding_distance = get_crowding_distance(population.fitness)

        population_size = len(population)
        tournament_indices = torch.randint(0, population_size, (selection_num, 2), device=population.device)

        ind1_indices = tournament_indices[:, 0]
        ind2_indices = tournament_indices[:, 1]

        ind1_penalty = population.penalty[ind1_indices]
        ind2_penalty = population.penalty[ind2_indices]

        ind1_feasible = torch.all(ind1_penalty == 0, dim=1)
        ind2_feasible = torch.all(ind2_penalty == 0, dim=1)

        selected_indices = torch.empty(selection_num, dtype=torch.long, device=population.device)

        # Case 1: One feasible, one infeasible
        one_feasible_mask = ind1_feasible ^ ind2_feasible
        selected_indices[one_feasible_mask] = torch.where(
            ind1_feasible[one_feasible_mask],
            ind1_indices[one_feasible_mask],
            ind2_indices[one_feasible_mask]
        )

        # Case 2: Both feasible
        both_feasible_mask = ind1_feasible & ind2_feasible
        ind1_front_rank = front_rank[ind1_indices[both_feasible_mask]]
        ind2_front_rank = front_rank[ind2_indices[both_feasible_mask]]

        front_rank_diff_mask = ind1_front_rank != ind2_front_rank
        selected_indices[both_feasible_mask][front_rank_diff_mask] = torch.where(
            ind1_front_rank[front_rank_diff_mask] < ind2_front_rank[front_rank_diff_mask],
            ind1_indices[both_feasible_mask][front_rank_diff_mask],
            ind2_indices[both_feasible_mask][front_rank_diff_mask]
        )

        same_front_mask = ~front_rank_diff_mask
        selected_indices[both_feasible_mask][same_front_mask] = torch.where(
            crowding_distance[ind1_indices[both_feasible_mask][same_front_mask]] > crowding_distance[ind2_indices[both_feasible_mask][same_front_mask]],
            ind1_indices[both_feasible_mask][same_front_mask],
            ind2_indices[both_feasible_mask][same_front_mask]
        )

        # Case 3: Both infeasible
        both_infeasible_mask = ~ind1_feasible & ~ind2_feasible
        ind1_total_penalty = torch.sum(population.penalty[ind1_indices[both_infeasible_mask]], dim=1)
        ind2_total_penalty = torch.sum(population.penalty[ind2_indices[both_infeasible_mask]], dim=1)
        selected_indices[both_infeasible_mask] = torch.where(
            ind1_total_penalty < ind2_total_penalty,
            ind1_indices[both_infeasible_mask],
            ind2_indices[both_infeasible_mask]
        )

        return population.index_select(selected_indices)