import torch
from torch.nn.functional import conv2d

from model.quanCNN import quanCNN

from .quantizer_torch import BiasScaleQuantizer, WeightQuantizer

import numpy as np

class inferCNN(torch.nn.Module):

    def __init__(self, originNN: quanCNN, M: int) -> None:
        super(inferCNN, self).__init__()
        
        self.conv1Params = (originNN.conv1.bias, originNN.conv1.padding, originNN.conv1.stride)
        self.sw1 = originNN.quan_weight1.alpha.detach()
        self.quanWeight1 = WeightQuantizer(self.sw1,
                                            upper_bound=originNN.quan_weight1.upper_bound,
                                            lower_bound=originNN.quan_weight1.lower_bound)
        self.weightConv1 = torch.nn.Parameter(self.quanWeight1(originNN.conv1.weight.detach()), requires_grad=False)

        self.conv2Params = (originNN.conv2.bias, originNN.conv2.padding, originNN.conv2.stride)
        self.sw2 = originNN.quan_weight2.alpha.detach()
        self.quanWeight2 = WeightQuantizer(self.sw2,
                                            upper_bound=originNN.quan_weight2.upper_bound,
                                            lower_bound=originNN.quan_weight2.lower_bound)
        self.weightConv2 = torch.nn.Parameter(self.quanWeight2(originNN.conv2.weight.detach()), requires_grad=False)

        self.saInput = originNN.quan_input.alpha.detach()
        self.sa1 = originNN.quan_act1.alpha.detach()
        self.sa2 = originNN.quan_act2.alpha.detach()

        self.quanAct1 = BiasScaleQuantizer(upper_bound=originNN.quan_act1.upper_bound,
                                            lower_bound=originNN.quan_act1.lower_bound,
                                            preScale=self.saInput,
                                            weightScale=self.sw1,
                                            scale=self.sa1,
                                            M=M)
        self.quanAct2 = BiasScaleQuantizer(upper_bound=originNN.quan_act2.upper_bound,
                                            lower_bound=originNN.quan_act2.lower_bound,
                                            preScale=self.sa1,
                                            weightScale=self.sw2,
                                            scale=self.sa2,
                                            M=M)
        
        self.swfc = originNN.quan_fc_weight.alpha.detach()
        self.quanFcWeight = WeightQuantizer(self.swfc,
                                            upper_bound=originNN.quan_fc_weight.upper_bound,
                                            lower_bound=originNN.quan_fc_weight.lower_bound)
        self.weightFc = torch.nn.Parameter(self.quanFcWeight(originNN.fc1.weight.detach()), requires_grad=False)
        self.quanFcBias = WeightQuantizer(self.swfc * self.sa2,
                                            upper_bound=originNN.quan_fc_weight.upper_bound,
                                            lower_bound=originNN.quan_fc_weight.lower_bound)
        self.biasFc = torch.nn.Parameter(self.quanFcBias(originNN.fc1.bias.detach()), requires_grad=False)


    def forward(self, x):

        x = x / self.saInput
        x = torch.clamp(x, -128, 127)
        x = x.round()


        #act_dict = {}
        #act_name = "input"
        #act_dict[act_name] = x.detach().cpu().numpy()

        x = conv2d(x, self.weightConv1, bias=self.conv1Params[0], padding=self.conv1Params[1], stride=self.conv1Params[2])
        x = self.quanAct1(x)

        #act_name = "conv1"
        #act_dict[act_name] = x.detach().cpu().numpy()

        x = conv2d(x, self.weightConv2, bias=self.conv2Params[0], padding=self.conv2Params[1], stride=self.conv2Params[2])
        x = self.quanAct2(x)

        #act_name = "conv2"
        #act_dict[act_name] = x.detach().cpu().numpy()

        x = x.view(x.size(0), -1)
        x = torch.nn.functional.linear(x, self.weightFc, self.biasFc)

        #act_name = "fc"
        #act_dict[act_name] = x.detach().cpu().numpy()

        #save_path = "./inferSim/outputs/sim_act.npz"
        #np.savez(save_path, **act_dict)

        return x