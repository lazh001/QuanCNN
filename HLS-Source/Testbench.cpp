#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ap_int.h>

typedef ap_int<8> data_t;
typedef ap_int<16> acc_t;

// 声明 HLS 模块接口
void conv_accel(data_t *In, data_t *W, acc_t *Out, int R, int C, int CHIn, int CHOut, int K, int S);

int main() {
    // 动态输入大小的测试用例
    int R = 32;       // Output rows (减小以测试内存问题)
    int C = 32;       // Output columns
    int K = 3;        // Kernel size
    int CHIn = 4;     // Input channels
    int CHOut = 4;    // Output channels
    int stride = 1;   // Stride

    // 计算输入和输出的实际尺寸
    int Rin = R + K - 1; // Padded input rows
    int Cin = C + K - 1; // Padded input columns

    // 使用固定大小的数组代替动态分配
    data_t In[CHIn * Rin * Cin];
    data_t W[CHOut * CHIn * K * K];
    acc_t Out[CHOut * R * C];
    acc_t Out_golden[CHOut * R * C];


    // 检查是否成功分配内存
    if (!In || !W || !Out || !Out_golden) {
        std::cerr << "Memory allocation failed. Reduce array sizes." << std::endl;
        return -1;
    }

    // 初始化输入特征图和权重
    for (int i = 0; i < CHIn * Rin * Cin; i++) {
        In[i] = static_cast<data_t>(rand() % 10);  // 随机生成 0-9 的整数
    }

    for (int i = 0; i < CHOut * CHIn * K * K; i++) {
        W[i] = static_cast<data_t>(rand() % 10);  // 随机生成 0-9 的整数
    }

    // 初始化输出和参考结果
    for (int i = 0; i < CHOut * R * C; i++) {
        Out[i] = 0;
        Out_golden[i] = 0;
    }

    // 使用 CPU 实现的参考卷积结果（黄金结果）
    for (int cho = 0; cho < CHOut; cho++) {
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                for (int chi = 0; chi < CHIn; chi++) {
                    for (int kr = 0; kr < K; kr++) {
                        for (int kc = 0; kc < K; kc++) {
                            if ((r + kr) < Rin && (c + kc) < Cin) {  // 检查索引
                                Out_golden[cho * R * C + r * C + c] +=
                                    W[cho * CHIn * K * K + chi * K * K + kr * K + kc] *
                                    In[chi * Rin * Cin + (r + kr) * Cin + (c + kc)];
                            }
                        }
                    }
                }
            }
        }
    }

    // 调用 HLS 模块
    std::cout << "Calling HLS conv_accel function..." << std::endl;
    conv_accel(In, W, Out, R, C, CHIn, CHOut, K, stride);
    std::cout << "HLS conv_accel called finished." << std::endl;

    // 验证 HLS 模块输出与参考结果
    bool pass = true;
    for (int i = 0; i < CHOut * R * C; i++) {
        if (Out[i] != Out_golden[i]) {  // 比较整数类型，无需浮点误差范围
            std::cout << "Mismatch at index " << i
                      << ": HLS Output = " << Out[i]
                      << ", Golden Output = " << Out_golden[i] << std::endl;
            pass = false;
            break;
        }
    }

    // 打印结果
    if (pass) {
        std::cout << "Test Passed: HLS Output matches Golden Reference." << std::endl;
    } else {
        std::cout << "Test Failed: HLS Output does not match Golden Reference." << std::endl;
    }

    // 清理动态分配的内存
    // delete[] In;
    // delete[] W;
    // delete[] Out;
    // delete[] Out_golden;

    return 0;
}
