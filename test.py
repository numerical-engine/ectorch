# import ectorch
# import torch
# import torch.nn as nn
# import sys

# class Net(nn.Module):
#     def forward(self, x):
#         return x**2

# class Sphere(ectorch.Function):
#     def convert(self, individuals:torch.Tensor)->torch.Tensor:
#         return individuals["R"]
    
#     def forward(self, *outputs:torch.Tensor)->torch.Tensor:
#         return torch.sum(outputs[0], dim=1)

# class Env(ectorch.Environment):
#     def __init__(self, function:ectorch.Function)->None:
#         super().__init__(functions=[function], penalty_functions=None)
    
#     def get_score(self, population:"Population")->torch.Tensor:
#         return torch.sum(population.fitness, dim = 1)


# population_num = 10
# vars = {
#     "R": torch.randn(population_num, 2),
#     "B": torch.randint(0, 2, (population_num, 2), dtype=torch.float32),
#     "Z": torch.randint(-2, 3, (population_num, 2), dtype=torch.float32),
# }

# population = ectorch.Population(variables=vars)
# population.to("cuda")
# net = Net().to("cuda")
# function = Sphere(net=net)
# environment = Env(function=function)
# environment.run(population)
# population.sort()
# print("Score:\n", population.score)

import torch
import ectorch
from matplotlib import pyplot as plt

fitness = 8.*(torch.rand(100, 2, device = "cuda")- 0.5)
# dominated_map = ectorch.utils.score_utils.get_dominated_map(fitness)
front_rank = ectorch.utils.score_utils.get_crowding_distance(fitness)
print(front_rank)


# Scattering plot of fitness with rank as color
plt.scatter(fitness[:,0].cpu(), fitness[:,1].cpu(), c=front_rank.cpu(), cmap='viridis')
plt.colorbar(label='Rank (Number of Dominating Individuals)')
plt.xlabel('Fitness Objective 1')
plt.ylabel('Fitness Objective 2')
plt.title('Fitness Scatter Plot Colored by Rank')
plt.savefig('fitness_rank_scatter.png')