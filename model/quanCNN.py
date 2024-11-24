import torch
import torch.nn as nn
import math

from neuralzip.quantizer.lsq import LearnedStepQuantizer

class quanCNN(nn.Module):

    def __init__(self, num_classes = 10,
                 quan_input = {"bit": 8, "all_positive": False, "symmetric": False},
                 quan_act1 = {"bit": 8, "all_positive": True, "symmetric": False},
                 quan_act2 = {"bit": 8, "all_positive": True, "symmetric": False},
                 quan_weight = {"bit": 8, "all_positive": False, "symmetric": False},
                 quan_fc_weight = {"bit": 8, "all_positive": False, "symmetric": False},):
        super(quanCNN, self).__init__()
        # input size: 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 4, kernel_size = 5, stride = 2, padding = 0, bias=False)
        self.conv2 = nn.Conv2d(4, 8, kernel_size = 3, stride = 2, padding = 0, bias=False)
        self.fc1 = nn.Linear(8 * 5 * 5, num_classes)
        self.relu = nn.ReLU(inplace = True)

        self.quan_input = LearnedStepQuantizer(**quan_input)
        self.quan_act1 = LearnedStepQuantizer(**quan_act1)
        self.quan_act2 = LearnedStepQuantizer(**quan_act2)   
        self.quan_weight1 = LearnedStepQuantizer(**quan_weight)
        self.quan_weight2 = LearnedStepQuantizer(**quan_weight)
        self.quan_fc_weight = LearnedStepQuantizer(**quan_fc_weight)

    def forward(self, x):
        x = self.quan_input(x)
        x = self.conv1._conv_forward(x, self.quan_weight1(self.conv1.weight), self.conv1.bias)
        x = self.quan_act1(x)
        x = self.relu(x)

        x = self.conv2._conv_forward(x, self.quan_weight2(self.conv2.weight), self.conv2.bias)
        x = self.quan_act2(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        x = torch.nn.functional.linear(x, self.quan_fc_weight(self.fc1.weight), self.fc1.bias)
        return x

def quanCnnObject(num_classes = 10, pretrained = None):
    model = quanCNN(num_classes = num_classes)
    
    if pretrained:
        if pretrained.endswith(".pth"):
            checkpoint = torch.load(pretrained)
            model.load_state_dict(checkpoint['model'], strict=False)
        elif pretrained.endswith(".ckpt"):
            checkpoint = torch.load(pretrained)
            model.load_state_dict({k.replace('model.',''):v for k,v in checkpoint['state_dict'].items()}, strict=False)
    return model