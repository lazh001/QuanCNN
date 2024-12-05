# 加速器的硬件实现

C++的接口如下

```c++
void conv_accel(
    data_t *In,    // Input feature map
    data_t *W,     // Weights
    acc_t *Out,    // Output feature map
    int R,         // Rows of the input feature map
    int C,         // Columns of the input feature map
    int CHIn,      // Input channels
    int CHOut,     // Output channels
    int K,         // Kernel size
    int S          // Stride
) 
```



其中使用的接口：

```c++
    #pragma HLS INTERFACE m_axi depth=1024 port=In offset=slave bundle=bus1
    #pragma HLS INTERFACE m_axi depth=1024 port=W offset=slave bundle=bus2
    #pragma HLS INTERFACE m_axi depth=1024 port=Out offset=slave bundle=bus1

    #pragma HLS INTERFACE mode=ap_none port=In
    #pragma HLS INTERFACE mode=ap_none port=W
    #pragma HLS INTERFACE mode=ap_none port=Out
```

涉及到数据的部分使用的是AXI接口



```c++
    #pragma HLS INTERFACE s_axilite port=R bundle=control
    #pragma HLS INTERFACE s_axilite port=C bundle=control
    #pragma HLS INTERFACE s_axilite port=CHIn bundle=control
    #pragma HLS INTERFACE s_axilite port=CHOut bundle=control
    #pragma HLS INTERFACE s_axilite port=K bundle=control
    #pragma HLS INTERFACE s_axilite port=S bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
```

参数部分使用lite协议



调用：

```
conv_accel(In, W, Out, R, C, CHIn, CHOut, K, stride);
```

详细代码参考`Testbench.cpp`