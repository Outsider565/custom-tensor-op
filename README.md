# 定制一个新的张量运算

复旦大学 王少文 王艺坤

## 实验目的

1. 理解DNN框架中的张量算子的原理

2. 基于不同方法实现新的张量运算，并比较性能差异

## 实验原理

1. 深度神经网络中的张量运算原理

2. PyTorch中基于Function和Module构造张量的方法

3. 通过C++扩展编写Python函数模块

### 实验环境

| 环境           | 具体型号/版本                                |
| ------------ | -------------------------------------- |
| CPU          | AMD Ryzen 7 5800H                      |
| GPU          | NVIDIA GeForce RTX 3070 Mobile / Max-Q |
| OS版本         | Ubuntu 21.04 x86_64                    |
| Python包名称及版本 | Python 3.8.10,pytorch 1.10.0           |
| CUDA版本       | cuda_11.2                              |

### 实验步骤

1. 在MNIST的模型样例中，选择线性层（Linear）张量运算进行定制化实现

2. 理解PyTorch构造张量运算的基本单位：Function和Module

3. 基于Function和Module的Python API重新实现Linear张量运算
   
   1. 修改MNIST样例代码
   
   2. 基于PyTorch  Module编写自定义的Linear 类模块
   
   3. 基于PyTorch Function实现前向计算和反向传播函数
   
   4. 使用自定义Linear替换网络中nn.Linear() 类
   
   5. 运行程序，验证网络正确性

4. 理解PyTorch张量运算在后端执行原理

5. 实现C++版本的定制化张量运算
   
   1. 基于C++，实现自定义Linear层前向计算和反向传播函数，并绑定为Python模型
   
   2. 将代码生成python的C++扩展
   
   3. 使用基于C++的函数扩展，实现自定义Linear类模块的前向计算和反向传播函数
   
   4. 运行程序，验证网络正确性

6. 使用profiler比较网络性能：比较原有张量运算和两种自定义张量运算的性能

7. 【可选实验，加分】实现卷积层（Convolutional）的自定义张量运算

### 实验记录

1. Linear层的实现

我实现的linear层相较于参考代码额外加入了`bias`

```python
class myLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(myLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(input_features,output_features))
        self.bias = torch.nn.Parameter(torch.randn(output_features))

    def forward(self, input):
        return myLinearFunction.apply(input, self.weight, self.bias)
```

2. MyLinearFunction的实现

```python
class myLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_val, weight, bias):
        ctx.save_for_backward(input_val, weight)
        return input_val.mm(weight) + bias.unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        input_val, weight = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight.t())
        if ctx.needs_input_grad[1]:
            grad_weight = input_val.t().mm(grad_output)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_biaseight, grad_bias
```

3. C++版本的实现

在mylinear.cpp中定义如下两个函数并进行绑定

```cpp
#include <torch/extension.h>

#include <iostream>
#include <vector>

std::vector<torch::Tensor> mylinear_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias)
{
    auto output = torch::mm(input, weights) + bias.unsqueeze(0);

    return {output};
}

std::vector<torch::Tensor> mylinear_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights
    ) 
{
    auto grad_input = torch::mm(grad_output, weights.t());
    auto grad_weights = torch::mm(input.t(), grad_output);
    auto grad_bias = torch::sum(grad_output, 0);

    return {grad_input, grad_weights, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mylinear_forward, "myLinear forward");
  m.def("backward", &mylinear_backward, "myLinear backward");
}
```

然后在python中调用即可

```python
class myLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_val, weight, bias):
        ctx.save_for_backward(input_val, weight)
        output = mylinear_cpp.forward(input_val, weight, bias)
        return output[0]
    
    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        input_val, weight = ctx.saved_variables
        grad_input, grad_weight, grad_bias = mylinear_cpp.backward(grad_output, input_val, weight)
        return grad_input, grad_weight, grad_bias
```

运行`python setup.py install`安装`mylinear_cpp_extension`

### 网络性能正确性

将修改后的Linear分别放入`mnist_custom_linear.py`和`mnist_custom_linear_cpp.py`，网络仍然能正确拟合。

### 性能评测

测试代码如下：

```python
import torch
import torch.nn as nn
import time
import mylinear_cpp

INPUT_SIZE = 1024
OUTPUT_SIZE = 128
TIMES = 10000


class myLinearFunctionCpp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_val, weight, bias):
        ctx.save_for_backward(input_val, weight)
        output = mylinear_cpp.forward(input_val, weight, bias)
        return output[0]
    
    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        input_val, weight = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = mylinear_cpp.backward(grad_output, input_val, weight)
        return grad_input, grad_weight, grad_bias

class myLinearCpp(nn.Module):
    def __init__(self, input_features, output_features):
        super(myLinearCpp, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(input_features,output_features))
        self.bias = torch.nn.Parameter(torch.randn(output_features))
        
    def forward(self, input):
        return myLinearFunctionCpp.apply(input, self.weight, self.bias)

class myLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_val, weight, bias):
        ctx.save_for_backward(input_val, weight)
        return input_val.mm(weight) + bias.unsqueeze(0)
    
    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        input_val, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight.t())
        if ctx.needs_input_grad[1]:
            grad_weight = input_val.t().mm(grad_output)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias
    
class myLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(myLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(input_features,output_features))
        self.bias = torch.nn.Parameter(torch.randn(output_features))
        
    def forward(self, input):
        return myLinearFunction.apply(input, self.weight, self.bias)
    
def run_once_with_grad(net: nn.Module):
    input_val = torch.randn(1,INPUT_SIZE)
    result = net(input_val)
    loss = torch.sum(result)
    loss.backward()
    return loss

def run_once(net: nn.Module):
    input_val = torch.randn(1,INPUT_SIZE)
    result = net(input_val)
    loss = torch.sum(result)
    return loss

def perf_forward(net: nn.Module):
    with torch.no_grad():
        t_start = time.time()
        for i in range(TIMES):
            run_once(net)
        t_end = time.time()
        return (t_end - t_start) / TIMES

def perf_overall(net: nn.Module):
    t_start = time.time()
    for i in range(TIMES):
        run_once_with_grad(net)
    t_end = time.time()
    return (t_end - t_start) / TIMES

if __name__ == '__main__':
    torch.manual_seed(42)
    linear = nn.Linear(INPUT_SIZE, OUTPUT_SIZE)
    mylinear = myLinear(INPUT_SIZE, OUTPUT_SIZE)
    mylinear_cc = myLinearCpp(INPUT_SIZE, OUTPUT_SIZE)
    print("Forward:")
    print("Pytorch:", perf_forward(linear))
    print("MyLinear:", perf_forward(mylinear))
    print("MyLinearCpp:", perf_forward(mylinear_cc))
    print("Overall:")
    print("Pytorch:", perf_overall(linear))
    print("MyLinear:", perf_overall(mylinear))
    print("MyLinearCpp:", perf_overall(mylinear_cc))
    
```

测试结果如下：

|                 | Forward | Forward+Backward |
| --------------- | ------- | ---------------- |
| Pytorch         | 7.7e-5  | 4.6e-4           |
| Mylinear+Python | 8.75e-5 | 4.5e-4           |
| Mylinear+Cpp    | 8.54e-5 | 5.6e-4           |

性能上而言`Pytorch`和`Mylinear+Python`较为优秀，可能是因为Linear层计算量并不大，而Python与C++之间的API调用反而给`Mylinear+Cpp`带来了额外开销。

### 拓展部分

自己实现一个卷积神经网络

```python
class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        '''
        In my simplified version, the stride is always 1 and the padding is always 0
        '''
        super(MyConv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size))
        torch.nn.init.xavier_uniform_(self.weight) # Important! Easier to fit the data
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, input_val: torch.Tensor) -> torch.Tensor:
        '''
        input_val: [batch_size, in_channels, height, width]
        return: [batch_size, out_channels, height_o, width_o]
        '''
        output_val = F.conv2d(input_val, self.weight, self.bias)
        return output_val
```

运行网络，仍然能正确拟合。
