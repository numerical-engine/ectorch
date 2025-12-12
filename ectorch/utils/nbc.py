import torch
import torch_geometric
import networkx as nx
import sys
import numpy as np

def nearest_better_clustering(points:torch.Tensor, score:torch.Tensor, phi:float = 2.0)->list[torch.Tensor]:
    """Nearest Better Clustering (NBC) algorithm to cluster points based on their distances.

    Args:
        points (torch.Tensor): A tensor of shape (num_points, num_dimensions) representing the coordinates of points to be clustered.
        score (torch.Tensor): A tensor of shape (num_points,) representing the score of each point.
        phi (float, optional): Cutting edge parameter controlling the clustering sensitivity. Defaults to 2.0.

    Returns:
        list[torch.Tensor]: List of tensors. length of the list is the number of clusters,
                            each tensor contains the indices of points belonging to that cluster.
    """
    disconnected_threshold = 1e10 # Use flag value to indicate disconnection
    num_points = points.size(0)
    
    # Compute distance matrix
    dist_matrix = torch.cdist(points, points, p=2)  # Shape: (num_points, num_points)
    dist_matrix += torch.eye(num_points, device=points.device) * disconnected_threshold  # Avoid zero distance to self
    
    # Set dist_matrix[i][j] zero if score[i] >= score[j]
    score_matrix = score.unsqueeze(1).repeat(1, num_points)  # Shape : (num_points, num_points)
    inf_mask = score_matrix >= score_matrix.t()
    dist_matrix[inf_mask] = disconnected_threshold
    
    # Get nearest index with ignoring zero distances
    row = torch.arange(num_points, device=points.device)
    col = torch.argmin(dist_matrix, dim=1)
    distance = dist_matrix[row, col]
    
    # remove index "i" from row and col if col[i] == disconnected_threshold
    valid_mask = distance < disconnected_threshold
    row = row[valid_mask]
    col = col[valid_mask]
    distance = distance[valid_mask]

    # Apply cut-off distance
    mean_distance = torch.mean(distance)
    cut_off_distance = phi * mean_distance
    cut_off_mask = distance <= cut_off_distance
    row = row[cut_off_mask]
    col = col[cut_off_mask]

    # Define graph
    edge_index = torch.stack([row, col], dim=0)  # Shape: (2, num_edges)
    graph = torch_geometric.data.Data(x=points, edge_index=edge_index)
    # Transform graph to undirected graph
    undirected_graph = torch_geometric.utils.to_undirected(graph.edge_index)
    graph.edge_index = undirected_graph
    #Convert to networksx graph
    graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    graphs = nx.components.connected_components(graph)

    clustes = [tuple(g) for g in graphs]
    
    return clustes


def nearest_better_clustering2(points:torch.Tensor, score:torch.Tensor, phi:float = 2.0, b:float = None)->list[torch.Tensor]:
    """Nearest Better Clustering (NBC) algorithm to cluster points based on their distances.

    Args:
        points (torch.Tensor): A tensor of shape (num_points, num_dimensions) representing the coordinates of points to be clustered.
        score (torch.Tensor): A tensor of shape (num_points,) representing the score of each point.
        phi (float, optional): Cutting edge parameter controlling the clustering sensitivity. Defaults to 2.0.
        b (float, optional): Cutting edge parameter controlling the clustering sensitivity. Defaults to None.

    Returns:
        list[torch.Tensor]: List of tensors. length of the list is the number of clusters,
                            each tensor contains the indices of points belonging to that cluster.
    """
    disconnected_threshold = 1e10 # Use flag value to indicate disconnection
    num_points = points.size(0)
    dim = points.size(1)
    if b is None:
        b = (-4.69e-4*dim**2 + 0.0263*dim + 3.66/dim - 0.457)*np.log10(num_points) + 7.51e-4*dim**2 - 0.0421*dim - 2.26/dim + 1.83
    
    # Compute distance matrix
    dist_matrix = torch.cdist(points, points, p=2)  # Shape: (num_points, num_points)
    dist_matrix += torch.eye(num_points, device=points.device) * disconnected_threshold  # Avoid zero distance to self
    
    # Set dist_matrix[i][j] zero if score[i] >= score[j]
    score_matrix = score.unsqueeze(1).repeat(1, num_points)  # Shape : (num_points, num_points)
    inf_mask = score_matrix >= score_matrix.t()
    dist_matrix[inf_mask] = disconnected_threshold
    
    # Get nearest index
    row = torch.arange(num_points, device=points.device)
    col = torch.argmin(dist_matrix, dim=1)
    distance = dist_matrix[row, col]
    
    # remove index "i" from row and col if col[i] == disconnected_threshold
    valid_mask = distance < disconnected_threshold
    row = row[valid_mask]
    col = col[valid_mask]
    distance = distance[valid_mask]

    # Apply cut-off distance
    mean_distance = torch.mean(distance)
    cut_off_distance = phi * mean_distance
    cut_off_mask = distance <= cut_off_distance
    row = row[cut_off_mask]
    col = col[cut_off_mask]
    distance = distance[cut_off_mask]
    
    # Define directed adjacency matrix
    adjacency_matrix = torch.zeros((num_points, num_points), device=points.device)
    adjacency_matrix[row, col] = distance
    
    # Cut edges based on b parameter
    num_outedge = torch.sum(adjacency_matrix > 0, dim=1)  # Shape: (num_points,)
    num_inedge = torch.sum(adjacency_matrix > 0, dim=0)  # Shape: (num_points,)

    mean_distance_in = torch.sum(adjacency_matrix, dim = 0) / (num_inedge + 1e-10)
    distance_out = torch.sum(adjacency_matrix, dim = 1) / (num_outedge + 1e-10)

    mask = (
        (num_outedge > 0.0)
        * (num_inedge > 2)
        * (distance_out / mean_distance_in > b)
    )
    adjacency_matrix[mask, :] = 0.0

    # Convert COO format
    row, col = torch.nonzero(adjacency_matrix, as_tuple=True)
    adjacency_matrix = torch.stack([row, col], dim=0)

    # Define graph
    graph = torch_geometric.data.Data(x=points, edge_index=adjacency_matrix)
    # Transform graph to undirected graph
    undirected_graph = torch_geometric.utils.to_undirected(graph.edge_index)
    graph.edge_index = undirected_graph
    #Convert to networksx graph
    graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    graphs = nx.components.connected_components(graph)

    clustes = [tuple(g) for g in graphs]
    
    return clustes