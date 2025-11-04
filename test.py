import torch
import torch.nn as nn
import ectorch as et

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 1)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x1, x2):
        return self.fc1(x1) + self.fc2(x2)

class Obj_func(et.Function):
    def forward(self, outputs:torch.Tensor)->torch.Tensor:
        return torch.sum(outputs, dim=1)
    
def converter(individuals:torch.Tensor):
    return individuals, individuals

net = SimpleModel()#.to('cuda')
individuals = torch.ones(100, 10).to('cuda')  # 100 individuals, each with 10 parameters
population = et.Population(individuals=individuals)
function = Obj_func(net=net, converter=converter)
function.to('cuda')
fitness = function(population)