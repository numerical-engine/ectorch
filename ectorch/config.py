import torch

class Config:
    """Class holding configuration parameters.

    All parameters are set by @property decorators. Therefore, they can not be modified.
    These parameters can be accessed by ectorch.config.<parameter_name>
    """
    @property
    def var_keys(self)->tuple[str]:
        """Returns the variable keys for the dataset.

        Returns:
            tuple[str]: A tuple containing the variable keys.
        """
        return (
            "R",  #Real
            "B",  #binary
            "Z",  #Integer
        )
    @property
    def dtype(self):
        return torch.float32

config = Config()