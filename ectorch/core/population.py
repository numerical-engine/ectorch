import torch

class Population:
    def __init__(self,
                individuals:torch.Tensor,
                generation:int = 0,
                age:torch.Tensor = None,
                fitness:torch.Tensor = None,
                penalty:torch.Tensor = None,
                score :torch.Tensor = None)->None:
        assert isinstance(individuals, torch.Tensor), "individuals must be a torch.Tensor"
        assert individuals.dim() == 2, "individuals must be a 2D tensor"
        
        self.individuals = individuals
        self.age = age if age is not None else torch.zeros(individuals.size(0), device=individuals.device)
        self.generation = generation
        self.fitness = fitness
        self.penalty = penalty
        self.score = score
    
    def reset(self)->None:
        self.fitness = None
        self.penalty = None
        self.score = None
    
    @property
    def device(self)->str:
        return self.individuals.device
    
    def to(self, device:str)->None:
        self.individuals = self.individuals.to(device)
        self.age = self.age.to(device)
        if self.fitness is not None:
            self.fitness = self.fitness.to(device)
        if self.penalty is not None:
            self.penalty = self.penalty.to(device)
        if self.score is not None:
            self.score = self.score.to(device)
    
    def __len__(self)->int:
        return self.individuals.size(0)
    def __getitem__(self, index:int)->torch.Tensor:
        return self.individuals[index]
    def __iter__(self):
        yield from self.individuals
    @property
    def already_eval(self)->bool:
        return (self.fitness is not None) and (self.penalty is not None) and (self.score is not None)
    @property
    def already_fit(self)->bool:
        return (self.fitness is not None) and (self.penalty is not None)
    
    def copy(self)->"Population":
        return Population(
            individuals = self.individuals.clone(),
            generation = self.generation,
            age = self.age.clone(),
            fitness = self.fitness.clone() if self.fitness is not None else None,
            penalty = self.penalty.clone() if self.penalty is not None else None,
            score = self.score.clone() if self.score is not None else None
        )