import torch
import torch.nn as nn

from ectorch.config import config

class Function:
    """Objective function class

    Return one fitness value for each individual in the population.

    Attributes:
        net (nn.Module): The neural network for the objective function.
    """
    def __init__(self, net:nn.Module)->None:
        self.net = net
        self.net.eval()
    
    def to(self, device:str)->None:
        """Moves the function network to the specified device.

        Args:
            device (str): The target device to move the function network to.
        """
        self.net.to(device)

    def convert(self, population:'Population')->torch.Tensor:
        """Converts the individuals tensor to the input format required by the network.

        Args:
            population (Population): The population object.
        Returns:
            torch.Tensor: The converted tensor suitable for the network input.
        Note:
            * This method should be overridden in subclasses.
        """
        raise NotImplementedError("The convert method should be implemented in subclasses.")
    
    def forward(self, *outputs:torch.Tensor)->torch.Tensor:
        """Computes the fitness from the network outputs.

        Returns:
            torch.Tensor: The computed fitness values of shape (population_size,).
        Note:
            * "The forward method should be implemented in subclasses."
        """
        raise NotImplementedError("The forward method should be implemented in subclasses.")
    
    def __call__(self, population:'Population')->torch.Tensor:
        """Evaluates the objective function for the given population.

        Args:
            population (Population): The population to be evaluated.
        Returns:
            torch.Tensor: The fitness tensor of shape (population_size,)
        """
        inputs = self.convert(population)
        with torch.no_grad():
            outputs = self.net(inputs)
        
        return self.forward(outputs)

class PenaltyFunction:
    """Penalty function class for handling constraints in the optimization process.

    Penalty functions are represented as self.criterion(x) < 0 for feasible regions.
    If the penalty value is greater than zero, it indicates that the individual violates certain constraints.
    """

    def to(self, device:str)->None:
        """Moves the penalty function network to the specified device.

        Args:
            device (str): The target device to move the penalty function network to.
        """
        pass

    def criterion(self, population:'Population')->torch.Tensor:
        """Computes the penalty criterion for the given population.

        The outputs tensor contains negative values which are converted to zero in __call__ method.

        Args:
            population (Population): The population to be evaluated.

        Returns:
            torch.Tensor: The computed penalty values of shape (population_size,).
        """
        raise NotImplementedError("The criterion method should be implemented in subclasses.")

    def __call__(self, population:'Population')->torch.Tensor:
        """Evaluates the penalty function for the given population.

        The responsibility of converting negative penalty values to zero lies in this method.

        Args:
            population (Population): The population to be evaluated.
        Returns:
            torch.Tensor: The penalty tensor of shape (population_size, 1)
        """
        penalties = self.criterion(population)
        penalties = torch.clamp(penalties, min = 0.0)
        penalties = penalties.unsqueeze(dim=1)
        return penalties