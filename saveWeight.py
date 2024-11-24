import torch as t
import numpy as np

from model import quanCnnObject
from inferSim import inferCNN

if __name__ == '__main__':
    ckpt_path = "./outputs/quanCnn/v2/checkpoints/epoch=9-val_loss_epoch=0.0832-val_acc_epoch=0.9763.ckpt"
    net = quanCnnObject(pretrained=ckpt_path)

    M = 9
    net_acc = inferCNN(originNN=net, M=M)

    weight_dict = {}
    weight_dict["saInput"] = net_acc.saInput.numpy()
    weight_dict["sw1"] = net_acc.sw1.numpy()
    weight_dict["sw2"] = net_acc.sw2.numpy()
    weight_dict["swfc"] = net_acc.swfc.numpy()
    weight_dict["sa1"] = net_acc.sa1.numpy()
    weight_dict["sa2"] = net_acc.sa2.numpy()
    weight_dict["weightConv1"] = net_acc.weightConv1.numpy()
    weight_dict["weightConv2"] = net_acc.weightConv2.numpy()
    weight_dict["weightFc"] = net_acc.weightFc.numpy()
    weight_dict["biasFc"] = net_acc.biasFc.numpy()

    save_path = "./inferSim/outputs/sim_weight.npz"
    np.savez(save_path, **weight_dict)