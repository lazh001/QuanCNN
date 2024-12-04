# QuanCNN

本项目实现了三个内容：网络训练，网络量化，推理仿真

## 网络训练
数据集使用MNIST，位于/dataset
网络定义位于/model/basicCNN.py
网络由两层卷积层，一层线性层组成（中间的激活层使用ReLU）
在main中使用basicCnn.yaml作为conf文件，即可进行网络训练
网络训练的log和权重位于/outputs/basicCnn

## 网络量化
使用LSQ算法进行量化感知训练QAT
主要实现方式是在网络中插入伪量化节点，将浮点数规范到整数的取值空间
量化训练的网络定义位于/model/quanCNN.py
在main中使用basicCnn.yaml作为conf文件，即可进行网络量化感知训练
量化感知训练的log和权重位于/outputs/quanCnn

## 推理仿真
本项目实现了一套能够模拟硬件推理过程的代码，在软硬件协同调试时，该代码可以给出硬件理想值，使调试过程更清晰明了。
推理仿真的输入数据位于/getFig。其中，mnist_npy文件采用npy格式，用于推理仿真程序的输入，mnist_fig文件采用txt格式，用于SDK调试程序的输入。
推理仿真的网络定义位于/inferSim，其中inferCNN.py仅进行推理仿真，inferCNN_saveAct.py在推理的同时会保存推理过程的中间激活值
推理仿真的主程序有两个，一个是singleInferSim.py，用于模拟单张图片的推理，并输出推理过程的中间激活值，另一个是saveWeight.py，用于输出模型量化后的权重
推理仿真的log位于/inferSim/logs，结果位于/inferSim/outputs
