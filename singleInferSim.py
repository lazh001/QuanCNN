import torch as t
import numpy as np

from model import quanCnnObject
from inferSim import inferCNN
from inferSim import inferCNN_saveAct


def arg_topK(matrix, K, axis=0):
    """
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    #full_sort = np.argsort(-matrix, axis=axis)
    #return full_sort.take(np.arange(K), axis=axis)
    topk = t.Tensor(matrix).topk(K, axis=axis)
    return topk.indices.numpy()

if __name__ == '__main__':

    ckpt_path = "./outputs/quanCnn/v2/checkpoints/epoch=9-val_loss_epoch=0.0832-val_acc_epoch=0.9763.ckpt"
    net = quanCnnObject(pretrained=ckpt_path)

    M = 9

    #net_acc = inferCNN(originNN=net, M=M)
    net_acc = inferCNN_saveAct(originNN=net, M=M)

    img_path = "./getFig/mnist_npy/5/2999.npy"
    input_data = np.load(img_path)
    print(input_data.shape)
    input_data = input_data.reshape(1, 1, 28, 28)

    input_data = t.Tensor(input_data).float()/255
    

    # test net with no grid
    net.eval()
    with t.no_grad():
        out_org = net(input_data)

    net_acc.eval()
    with t.no_grad():
        out, act_dict = net_acc(input_data)
        print(out.shape)



    out_topk = arg_topK(out, 5, axis=1)
    out_org_topk = arg_topK(out_org, 5, axis=1)
    print("out\n", out_topk)
    print("out_org\n",out_org_topk)



    with open("./inferSim/logs/outInfer.txt", "a+") as f:
        print("out\n", out_topk, file=f)
        print("out_org\n",out_org_topk, file=f)

    save_path = "./inferSim/outputs/sim_act.npz"
    np.savez(save_path, **act_dict)
    
    