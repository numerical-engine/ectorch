import torch

from ectorch.core.selection import Selection

class TopKSelection(Selection):
    """Select the top-k individuals from the population based on their scores.
    """
    def select(self, scores:torch.Tensor, selection_num:int)->torch.Tensor:
        _, indices = torch.topk(scores, selection_num, largest=True)
        return indices