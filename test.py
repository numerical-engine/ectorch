import ectorch as ec
import torch
import torch.nn as nn

class sphere(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return x**2

class SphereFunction(ec.Function):
    def convert(self, individuals:torch.Tensor)->tuple:
        return (individuals,)
    def forward(self, outputs:torch.Tensor)->torch.Tensor:
        return torch.sum(outputs, dim=1)

individuals = torch.randn(2, 3, device="cuda")  # 100 individuals, each with 3s features
population = ec.Population(individuals=individuals)
population2 = ec.Population(individuals=individuals)
populations = ec.utils.population.cat(population, population2)
print(populations.individuals)
print(population.individuals)
print(population2.individuals)