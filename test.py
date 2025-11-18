import ectorch
import torch
import torch.nn as nn
import sys

class Net(nn.Module):
    def forward(self, x):
        return x**2

class Sphere(ectorch.Function):
    def convert(self, individuals:torch.Tensor)->torch.Tensor:
        return individuals
    
    def forward(self, *outputs:torch.Tensor)->torch.Tensor:
        return torch.sum(outputs[0], dim=1)

population_num = 10
vars = {
    "R": torch.randn(population_num, 2),
    "B": torch.randint(0, 2, (population_num, 2), dtype=torch.float32),
    "Z": torch.randint(-2, 3, (population_num, 2), dtype=torch.float32),
}

population = ectorch.Population(variables=vars)
population.to("cuda")
net = Net().to("cuda")
function = Sphere(net=net)
xl = torch.tensor([-5.0, -5.0], device="cuda")
xu = torch.tensor([5.0, 5.0], device="cuda")
population.penalty = torch.zeros(population_num, 1, device="cuda")
population.fitness = torch.randn(population_num, 1, device="cuda")
population.score = population.fitness[:,0] + population.penalty[:,0]

selection = ectorch.selection.TournamentSelection()
print(population.score)
selected_population = selection(population, 2)
print(selected_population.score)