import torch
import torch.nn as nn

from ectorch.core.population import Population


class Function:
    """Base class for all functions.

    Attributes:
        net (nn.Module): The neural network to evaluate.
    """
    def __init__(self, net:nn.Module)->None:
        self.net = net
        self.net.eval()
    
    def to(self, device:str)->None:
        self.net.to(device)

    def __call__(self, population:Population)->torch.Tensor:
        """Return the fitness values of the population

        Args:
            population (Population): The population to evaluate.
        Returns:
            torch.Tensor: The fitness values of the population.
        """
        inputs = self.convert(population.individuals)
        with torch.no_grad():
            outputs = self.net(*inputs)
        
        return self.forward(outputs)
    
    def convert(self, individuals:torch.Tensor)->tuple:
        """Converts individuals to network inputs.

        Args:
            individuals (torch.Tensor): The individuals to convert. Shape is (N, M), where N is the number of individuals and M is the individual dimension.
        Returns:
            tuple: The converted inputs for the network.
        """
        raise NotImplementedError
    
    def forward(self, outputs:torch.Tensor)->torch.Tensor:
        """Compute the fitness values from the network outputs.

        Args:
            outputs (torch.Tensor): The outputs of the network. Shape is (N, D), where N is the number of individuals and D is the output dimension.
        Returns:
            torch.Tensor: The computed fitness values. Shape is (N,).
        """
        raise NotImplementedError


class PenaltyFunction:
    """Penalty function that penalizes individuals for being below a certain threshold.

    If the individual is feasible (i.e. criterion(individual) < 0), the penalty is zero.
    """
    def criterion(self, population:Population)->torch.Tensor:
        """Compute the penalty for individuals in the population.

        Args:
            population (Population): The population to evaluate.
        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        Returns:
            torch.Tensor: The computed penalties. Shape is (N,).
        """
        raise NotImplementedError
    
    def to(self, device:str)->None:
        pass
    
    def __call__(self, population:Population)->torch.Tensor:
        """Compute the penalty for the population.

        Args:
            population (Population): The population to evaluate.
        Returns:
            torch.Tensor: The computed penalties. Shape is (N,).
        """
        penalties = self.criterion(population)
        penalties = torch.clamp(penalties, min = 0.0)
        return penalties

class PenaltyLower(PenaltyFunction):
    """Penalty function that penalizes individuals for being below a certain threshold.

    If the individual is feasible (i.e. criterion(individual) < 0), the penalty is zero.
    """
    def __init__(self, xl:torch.Tensor)->None:
        self.xl = xl
    
    @property
    def device(self)->str:
        return self.xl.device
    
    def to(self, device:str)->None:
        self.xl = self.xl.to(device)

    def criterion(self, population:Population)->torch.Tensor:
        return self.xl - population.individuals

class PenaltyUpper(PenaltyFunction):
    """Penalty function that penalizes individuals for being above a certain threshold.

    If the individual is feasible (i.e. criterion(individual) < 0), the penalty is zero.
    """
    def __init__(self, xu:torch.Tensor)->None:
        self.xu = xu

    @property
    def device(self)->str:
        return self.xu.device
    
    def to(self, device:str)->None:
        self.xu = self.xu.to(device)

    def criterion(self, population:Population)->torch.Tensor:
        return population.individuals - self.xu