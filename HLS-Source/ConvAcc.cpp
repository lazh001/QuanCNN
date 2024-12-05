#include <ap_int.h>

typedef ap_int<8> data_t;  // 8-bit input data
typedef ap_int<16> acc_t;  // 16-bit accumulator for intermediate sums

// Local buffers (using fixed dimensions for array partitioning)
#define MAX_R 32
#define MAX_C 32
#define MAX_CHIN 4 
#define MAX_CHOUT 4
#define MAX_K 3

// Convolution Accelerator with AXI Interface
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
) {
    // printf("start conv_accel\n");
    // Define input and output sizes based on the provided R, C, K, and padding
    int Rin = R + K - 1;  // Input rows with padding
    int Cin = C + K - 1;  // Input columns with padding

    // Bind to AXI interfaces
    #pragma HLS INTERFACE m_axi depth=1024 port=In offset=slave bundle=bus1
    #pragma HLS INTERFACE m_axi depth=1024 port=W offset=slave bundle=bus2
    #pragma HLS INTERFACE m_axi depth=1024 port=Out offset=slave bundle=bus1

    // AXI-Lite control interface
    // #pragma HLS INTERFACE s_axilite port=In bundle=control
    // #pragma HLS INTERFACE s_axilite port=W bundle=control
    // #pragma HLS INTERFACE s_axilite port=Out bundle=control
    #pragma HLS INTERFACE mode=ap_none port=In
    #pragma HLS INTERFACE mode=ap_none port=W
    #pragma HLS INTERFACE mode=ap_none port=Out

    #pragma HLS INTERFACE s_axilite port=R bundle=control
    #pragma HLS INTERFACE s_axilite port=C bundle=control
    #pragma HLS INTERFACE s_axilite port=CHIn bundle=control
    #pragma HLS INTERFACE s_axilite port=CHOut bundle=control
    #pragma HLS INTERFACE s_axilite port=K bundle=control
    #pragma HLS INTERFACE s_axilite port=S bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control



    data_t In_local[MAX_CHIN][MAX_R + MAX_K - 1][MAX_C + MAX_K - 1];
    data_t W_local[MAX_CHOUT][MAX_CHIN][MAX_K][MAX_K];
    acc_t Out_local[MAX_CHOUT][MAX_R][MAX_C];  // No initialization

    #pragma HLS array_partition variable=In_local complete dim=1
    #pragma HLS array_partition variable=W_local complete dim=1
    #pragma HLS array_partition variable=W_local complete dim=2
    #pragma HLS array_partition variable=Out_local complete dim=1

    // initialize the input buffer to zero
    for (int chi = 0; chi < MAX_CHIN; chi++) {
        for (int r = 0; r < MAX_R + MAX_K - 1; r++) {
            for (int c = 0; c < MAX_C + MAX_K - 1; c++) {
                In_local[chi][r][c] = 0;
            }
        }
    }

    // initialize the weight buffer to zero
    for (int cho = 0; cho < MAX_CHOUT; cho++) {
        for (int chi = 0; chi < MAX_CHIN; chi++) {
            for (int kr = 0; kr < MAX_K; kr++) {
                for (int kc = 0; kc < MAX_K; kc++) {
                    W_local[cho][chi][kr][kc] = 0;
                }
            }
        }
    }

    // Load input feature map to local buffer
    Load_In: for (int chi = 0; chi < CHIn; chi++) {
        for (int r = 0; r < Rin; r++) {
            for (int c = 0; c < Cin; c++) {
                #pragma HLS PIPELINE
                In_local[chi][r][c] = In[chi * Rin * Cin + r * Cin + c];
            }
        }
    }

    // Load weights to local buffer
    Load_W: for (int cho = 0; cho < CHOut; cho++) {
        for (int chi = 0; chi < CHIn; chi++) {
            for (int kr = 0; kr < K; kr++) {
                for (int kc = 0; kc < K; kc++) {
                    #pragma HLS PIPELINE
                    W_local[cho][chi][kr][kc] = W[cho * CHIn * K * K + chi * K * K + kr * K + kc];
                }
            }
        }
    }

    // Initialize the output buffer to zero
    Initialize_Out: for (int cho = 0; cho < CHOut; cho++) {
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                #pragma HLS PIPELINE
                Out_local[cho][r][c] = 0;
            }
        }
    }

    // Compute convolution
    Kernel_Row: for (int kr = 0; kr < K; kr++) {
        Kernel_Column: for (int kc = 0; kc < K; kc++) {
            Row: for (int r = 0; r < R; r++) {
                Column: for (int c = 0; c < C; c++) {
                    Output_Channel: for (int cho = 0; cho < CHOut; cho++) {
                        #pragma HLS PIPELINE
                        acc_t acc = Out_local[cho][r][c]; // 初始化为当前累加值
                        Input_Channel: for (int chi = 0; chi < CHIn; chi++) {
                            #pragma HLS UNROLL
                            acc_t temp = (acc_t)W_local[cho][chi][kr][kc] * In_local[chi][S * r + kr][S * c + kc];
                            acc += temp;  // 定点数乘法累加
                        }
                        Out_local[cho][r][c] = acc; // 累加结果写回
                    }
                }
            }
        }
    }


    // Store output feature map back to external memory
    Store_Out: for (int cho = 0; cho < CHOut; cho++) {
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                #pragma HLS PIPELINE
                Out[cho * R * C + r * C + c] = Out_local[cho][r][c];
            }
        }
    }
}
