import torch
import torch.nn as nn
import time
import mylinear_cpp

INPUT_SIZE = 1024
OUTPUT_SIZE = 1024
TIMES = 100000


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
    
def run_once_with_grad(net: nn.Module, input_val: torch.Tensor = None):
    result = net(input_val)
    loss = torch.sum(result)
    loss.backward()
    return loss

def run_once(net: nn.Module, input_val: torch.Tensor = None):
    result = net(input_val)
    loss = torch.sum(result)
    return loss

def perf_forward(net: nn.Module):
    input_val = torch.randn(1,INPUT_SIZE)
    with torch.no_grad():
        t_start = time.time()
        for i in range(TIMES):
            run_once(net, input_val)
        t_end = time.time()
        return (t_end - t_start) / TIMES

def perf_overall(net: nn.Module):
    input_val = torch.randn(1,INPUT_SIZE)
    t_start = time.time()
    for i in range(TIMES):
        run_once_with_grad(net, input_val)
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
    