import torch

#****************************************************************************************
# Abstract class Quantizer
#****************************************************************************************
class Quantizer(torch.nn.Module):
    """Base class of quantizers."""

    def __init__(self) -> None:
        super().__init__()
        return

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    

#****************************************************************************************
# Quantizer for activation
#****************************************************************************************
class BiasScaleQuantizer(Quantizer):
    def __init__(self, upper_bound: int, lower_bound: int, preScale, weightScale, scale, M: int) -> None:
        super().__init__()
        self.upper_bound, self.lower_bound = upper_bound, lower_bound
        self.preScale = torch.nn.Parameter(preScale, requires_grad=False)
        self.weightScale = torch.nn.Parameter(weightScale, requires_grad=False)
        self.scale = torch.nn.Parameter(scale, requires_grad=False)
        self.M = M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = (x + self.bias).astype(np.int32) * self.gam.astype(np.int32)
        #result = x / (2**(self.M-1))
        #result = np.floor(result).astype(np.int32)
        #result = np.ceil(result / 2.0).astype(np.int32)
        #result = np.clip(result, self.lower_bound,  self.upper_bound).astype(np.int16)
        gam = torch.round(self.preScale * self.weightScale / self.scale * 2 ** (self.M))
        x = x * gam
        x = x / (2 ** (self.M - 1))
        x = torch.floor(x)
        x = torch.ceil(x / 2.0)
        x = torch.clip(x, self.lower_bound, self.upper_bound)
        return x
    
#****************************************************************************************
# Quantizer for weight
#****************************************************************************************
class WeightQuantizer(Quantizer):
    def __init__(self, scale: torch.Tensor, upper_bound: int, lower_bound: int) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(scale, requires_grad=False)
        self.upper_bound, self.lower_bound = upper_bound, lower_bound

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.round(x / self.scale)
        x = torch.clip(x, self.lower_bound, self.upper_bound)
        return x
