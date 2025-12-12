import torch
import itertools

def inclusion_exclusion(points:torch.Tensor, ref_point:torch.Tensor = None)->float:
    """Return hypervolume

    Args:
        points (torch.Tensor): Coordinate of all points with shape (Number of individuals, objective dimension).
        ref_point (torch.Tensor, optional): Reference point with shape (objective dimension,). Defaults to None.
    Returns:
        float: Hypervolume
    """
    if ref_point is None:
        ref_point = torch.max(points, dim=0).values
    N = len(points); D = len(ref_point)

    def volume_setprod(subset_point):
        max_point = torch.max(subset_point, dim=0).values
        bbox = ref_point - max_point
        assert torch.all((bbox >= 0.))
        return torch.prod(bbox)
    
    hv = 0.
    for r in range(1, N + 1):
        subsets = itertools.combinations(points, r)
        for subset in subsets:
            subset_points = torch.stack(subset, dim = 0)
            sign = (-1)**(r + 1)
            hv += sign*volume_setprod(subset_points)
    
    return hv.item()