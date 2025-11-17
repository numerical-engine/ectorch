import ectorch
import torch
import torch.nn as nn


class Net(nn.Module):
    def forward(self, x):
        return x**2

class Sphere(ectorch.Function):
    def convert(self, individuals:torch.Tensor)->torch.Tensor:
        return individuals
    
    def forward(self, *outputs:torch.Tensor)->torch.Tensor:
        return torch.sum(outputs[0], dim=1)

population_num = 3
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
mut = ectorch.mutation.real.PBMMutation(xl=xl, xu=xu, p=0.5, eta=20.0)
print(population.variables["R"])
new_population = mut(population)
print(new_population.variables["R"])