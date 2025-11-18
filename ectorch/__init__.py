"""ectorch: A library for evolutionary computation using PyTorch.
"""

from ectorch.core.population import Population
from ectorch.core.function import Function, PenaltyFunction
from ectorch.config import config
from ectorch.core.mutation import Mutation
from ectorch import mutation
from ectorch.core.crossover import Crossover
from ectorch import crossover
from ectorch.core.selection import Selection
from ectorch import selection

__version__ = "0.1.0"