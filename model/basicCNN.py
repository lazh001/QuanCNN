import torch
import torch.nn as nn

class basicCNN(nn.Module):

    def __init__(self, num_classes = 10):
        super(basicCNN, self).__init__()
        # input size: 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 4, kernel_size = 5, stride = 2, padding = 0, bias=False)
        self.conv2 = nn.Conv2d(4, 8, kernel_size = 3, stride = 2, padding = 0, bias=False)
        self.fc1 = nn.Linear(8 * 5 * 5, num_classes)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def basicCnnObject(num_classes = 10, pretrained = None):
    model = basicCNN(num_classes = num_classes)
    
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model