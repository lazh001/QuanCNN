import math

import torch

from .quantizer import Quantizer
from .helper import *

from .lsq import grad_scale, round_pass

#def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
#    y = x
#    y_grad = x * scale
#    return (y - y_grad).detach() + y_grad
#
#
#def round_pass(x: torch.Tensor) -> torch.Tensor:
#    y = x.round()
#    y_grad = x
#    return (y - y_grad).detach() + y_grad

class CompressedQuantizer(Quantizer):
    def __init__(self, bit: int, all_positive: bool = False, symmetric: bool = False, shiftbits: int = 9, bnLayer = None) -> None:
        super().__init__()
        self.upper_bound, self.lower_bound = quan_bound(bit, all_positive, symmetric)
        self.scale = torch.nn.Parameter(torch.tensor(1.))

        self.register_buffer('shiftbits', torch.tensor(shiftbits))

        # feature of the previous layer
        #self.preActScale = torch.nn.Parameter(torch.tensor(1.))
        #self.preWeightScale = torch.nn.Parameter(torch.tensor(1.))

        # features of the batch normalization layer
        if bnLayer is not None:
            self.register_buffer('bn_mean', bnLayer.running_mean)
            self.register_buffer('bn_var', bnLayer.running_var)
            self.eps = bnLayer.eps
            self.register_buffer('bn_w', bnLayer.weight)
            self.register_buffer('bn_b', bnLayer.bias)
        else:
            self.register_buffer('bn_mean', torch.zeros(1,))
            self.register_buffer('bn_var', torch.ones(1,))
            self.eps = 0
            self.register_buffer('bn_w', torch.ones(1,))
            self.register_buffer('bn_b', torch.zeros(1,))

        # flags to indicate whether Quantizer is initialized
        self.register_buffer("preScaleValid", torch.zeros(1))
        self.register_buffer('scaleValid', torch.zeros(1))

    def getPreScale(self, preActScale: torch.nn.Parameter, preWeightScale: torch.nn.Parameter):
        self.preActScale = preActScale
        self.preWeightScale = preWeightScale
        self.preScaleValid.fill_(1)


        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.scaleValid == 0:
            # initialize scale with the first batch of input data
            self.scale.data.copy_(2. * x.abs().mean() / math.sqrt(self.upper_bound))
            self.scaleValid.fill_(1)

        grad_scale_factor = math.sqrt(1. / (x.numel() * self.upper_bound))
        scale = grad_scale(self.scale, grad_scale_factor)
        #print(self.scale, self.preActScale, self.preWeightScale)
        if self.preScaleValid == 0:
            raise ValueError('Please call getPreScale first')

        preActScale = self.preActScale.detach()
        preWeightScale = self.preWeightScale.detach()

        bias = round_pass((-self.bn_mean + self.bn_b * torch.sqrt(self.bn_var + self.eps) / self.bn_w) / (preActScale * preWeightScale))
        gam = round_pass(preActScale * preWeightScale * self.bn_w / (torch.sqrt(self.bn_var + self.eps) * scale) * (2 ** self.shiftbits)) 
        bias = bias.reshape(1, -1, 1, 1)
        gam = gam.reshape(1, -1, 1, 1)
  
        x = x / (preActScale * preWeightScale)
        x = round_pass(((x + bias) * gam) / (2 ** self.shiftbits))
        x = x.clamp(min=self.lower_bound, max=self.upper_bound)
        x = x * scale
        return x

class TransWoConvQuantizer(Quantizer):
    def __init__(self, bit: int, all_positive: bool = False, symmetric: bool = False, shiftbits: int = 9) -> None:
        super().__init__()
        self.upper_bound, self.lower_bound = quan_bound(bit, all_positive, symmetric)

        self.register_buffer('shiftbits', torch.tensor(shiftbits))

        # flags to indicate whether Quantizer is initialized
        self.register_buffer("preScaleValid", torch.zeros(1))
        self.register_buffer('scaleValid', torch.zeros(1))

    def getPreScale(self, preActScale: torch.nn.Parameter):
        self.preActScale = preActScale
        self.preScaleValid.fill_(1)
    
    def getScale(self, scale: torch.nn.Parameter):
        self.scale = scale
        self.scaleValid.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.preScaleValid == 0 or self.scaleValid == 0:
            raise ValueError('Please call getPreScale and getScale first')
        
        scale = self.scale.detach()
        preActScale = self.preActScale.detach()
        bias = torch.zeros_like(x)
        gam = round_pass((preActScale / scale) * (2 ** self.shiftbits))
        gam = gam.reshape(1, -1, 1, 1)

        x = x / (preActScale)
        x = round_pass(((x + bias) * gam) / (2 ** self.shiftbits))
        x = x.clamp(min=self.lower_bound, max=self.upper_bound)
        x = x * scale
        return x

class TransWithConvQuantizer(Quantizer):
    def __init__(self, bit: int, all_positive: bool = False, symmetric: bool = False, shiftbits: int = 9, bnLayer = None) -> None:
        super().__init__()
        self.upper_bound, self.lower_bound = quan_bound(bit, all_positive, symmetric)

        self.register_buffer('shiftbits', torch.tensor(shiftbits))

        ## feature of the previous layer
        #self.preActScale = torch.nn.Parameter(torch.tensor(1.))
        #self.preWeightScale = torch.nn.Parameter(torch.tensor(1.))

        # features of the batch normalization layer
        if bnLayer is not None:
            self.register_buffer('bn_mean', bnLayer.running_mean)
            self.register_buffer('bn_var', bnLayer.running_var)
            self.eps = bnLayer.eps
            self.register_buffer('bn_w', bnLayer.weight)
            self.register_buffer('bn_b', bnLayer.bias)
        else:
            self.register_buffer('bn_mean', torch.zeros(1,))
            self.register_buffer('bn_var', torch.ones(1,))
            self.eps = 0
            self.register_buffer('bn_w', torch.ones(1,))
            self.register_buffer('bn_b', torch.zeros(1,))

        # flags to indicate whether Quantizer is initialized
        self.register_buffer("preScaleValid", torch.zeros(1))
        self.register_buffer('scaleValid', torch.zeros(1))

    def getPreScale(self, preActScale: torch.nn.Parameter, preWeightScale: torch.nn.Parameter):
        self.preActScale = preActScale
        self.preWeightScale = preWeightScale
        self.preScaleValid.fill_(1)

    def getScale(self, scale: torch.nn.Parameter):
        self.scale = scale
        self.scaleValid.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.preScaleValid == 0 or self.scaleValid == 0:
            raise ValueError('Please call getPreScale and getScale first')  

        scale = self.scale.detach()
        #scale = self.scale
        preActScale = self.preActScale.detach()
        preWeightScale = self.preWeightScale.detach()
        
        bias = round_pass((-self.bn_mean + self.bn_b * torch.sqrt(self.bn_var + self.eps) / self.bn_w) / (preActScale * preWeightScale))
        gam = round_pass(preActScale * preWeightScale * self.bn_w / (torch.sqrt(self.bn_var + self.eps) * scale) * (2 ** self.shiftbits)) 
        bias = bias.reshape(1, -1, 1, 1)
        gam = gam.reshape(1, -1, 1, 1)
  
        
        x = x / (preActScale * preWeightScale)
        x = round_pass(((x + bias) * gam) / (2 ** self.shiftbits))
        x = x.clamp(min=self.lower_bound, max=self.upper_bound)
        x = x * scale
        return x
    
class WeightQuantizer(Quantizer):
    def __init__(self, bit: int, all_positive: bool = False, symmetric: bool = False) -> None:
        super().__init__()
        self.upper_bound, self.lower_bound = quan_bound(bit, all_positive, symmetric)
        self.scale = torch.nn.Parameter(torch.tensor(1.))

        # a flag to indicate whether `scale` is initialized
        self.register_buffer('initialized', torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.initialized == 0:
            # initialize scale with the first batch of input data
            self.scale.data.copy_(2. * x.abs().mean() / math.sqrt(self.upper_bound))
            self.initialized.fill_(1)

        grad_scale_factor = math.sqrt(1. / (x.numel() * self.upper_bound))
        scale = grad_scale(self.scale, grad_scale_factor)

        x = round_pass(x / scale)
        x = x.clamp(min=self.lower_bound, max=self.upper_bound)
        x = x * scale
        return x

class BoundCheckQuantizer(Quantizer):
    def __init__(self, bit: int, all_positive: bool = False, symmetric: bool = False) -> None:
        super().__init__()
        self.upper_bound, self.lower_bound = quan_bound(bit, all_positive, symmetric)
                
        # flags to indicate whether Quantizer is initialized
        self.register_buffer('scaleValid', torch.zeros(1))
    
    def getScale(self, scale: torch.nn.Parameter):
        self.scale = scale
        self.scaleValid.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scaleValid != 0:
            scale = self.scale.detach()
        else:
            raise ValueError('Please call getScale first')
        x = x / scale
        x = x.clamp(min=self.lower_bound, max=self.upper_bound)
        x = x * scale
        return x