import torch
from torch import nn
import numpy as np

class myConv1d():
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # init parameters
        self.parameters = torch.randn((out_channels, in_channels, kernel_size), requires_grad=True)

        # init bias
        self.bias = torch.randn((out_channels, ), requires_grad=True)

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[0] == self.in_channels, 'in channels not match!'
        assert input.shape[1] >= self.kernel_size, 'input to less!'
        calculated = torch.zeros((self.out_channels, input.size(1) - self.kernel_size + 1))
        for i_out in range(self.out_channels):
            for i_in in range(self.in_channels):
                for i_w in range(calculated.shape[1]):
                    # print(i_out, i_in, i_w)
                    calculated[i_out][i_w] += torch.dot(self.parameters[i_out, i_in], input[i_in, i_w:i_w + self.kernel_size])
        calculated += self.bias.reshape((self.out_channels, 1))
        return calculated

class myConv2d():
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], stride: int = 1, bias: bool = True) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_bias = bias

        # init parameters
        self.parameters = torch.randn((self.out_channels, self.in_channels, *kernel_size), requires_grad=True)

        # init bias
        if bias:
            self.bias = torch.randn((self.out_channels, ), requires_grad=True)

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[0] == self.in_channels, 'in channels not match!'
        assert input.shape[1] >= self.kernel_size[0] and input.shape[2] >= self.kernel_size[1], 'input to less!'
        calculated = torch.zeros(self.out_channels, input.shape[1] - self.kernel_size[0] + 1, input.shape[2] - self.kernel_size[1] + 1)
        for i_out in range(calculated.shape[0]):
            for i_in in range(self.in_channels):
                for i_w in range(calculated.shape[1]):
                    for i_h in range(calculated.shape[2]):
                        calculated[i_out][i_w][i_h] += torch.sum(self.parameters[i_out, i_in] * input[i_in, i_w:i_w + self.kernel_size[0], i_h:i_h + self.kernel_size[1]])
        if self.use_bias:
            calculated += self.bias.reshape((self.out_channels, 1, 1))
        return calculated

    def zero_grad(self):
        self.parameters.grad.zero_()
        if self.use_bias:
            self.bias.grad.zero_()

class myConv3d():
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int, int], stride: int = 1, bias: bool = True) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_bias = bias

        # init parameters
        self.parameters = torch.randn((self.out_channels, self.in_channels, *kernel_size), requires_grad=True)

        # init bias
        if bias:
            self.bias = torch.randn((self.out_channels, ), requires_grad=True)
    
    # calculate conv3d use kernel Y on X
    @staticmethod
    def conv3d(X: torch.Tensor, Y: torch.Tensor, output: torch.Tensor):
        depth_cal = X.shape[0] - Y.shape[0] + 1
        width_cal = X.shape[1] - Y.shape[1] + 1
        height_cal = X.shape[2] - Y.shape[2] + 1
        for i_d in range(depth_cal):
            for i_w in range(width_cal):
                for i_h in range(height_cal):
                    output[i_d][i_w][i_h] += torch.sum(X[i_d:i_d + Y.shape[0], \
                                                         i_w:i_w + Y.shape[1], \
                                                         i_h:i_h + Y.shape[2]] * Y)

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[0] == self.in_channels, 'in channels not match!'
        assert input.shape[1] >= self.kernel_size[0] and input.shape[2] >= self.kernel_size[1] and input.shape[3] >= self.kernel_size[2], 'input to less!'
        calculated = torch.zeros(self.out_channels, input.shape[1] - self.kernel_size[0] + 1, \
                                                    input.shape[2] - self.kernel_size[1] + 1, \
                                                    input.shape[3] - self.kernel_size[2] + 1)
        for i_out in range(self.out_channels):
            for i_in in range(self.in_channels):
                self.conv3d(input[i_in], self.parameters[i_out, i_in], calculated[i_out])
        if self.use_bias:
            calculated += self.bias.reshape((self.out_channels, 1, 1, 1))
        return calculated

    def zero_grad(self):
        self.parameters.grad.zero_()
        if self.use_bias:
            self.bias.grad.zero_()

class myConv4d():
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int, int, int], stride: int = 1, bias: bool = True) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_bias = bias

        # init parameters
        self.parameters = torch.randn((self.out_channels, self.in_channels, *kernel_size), requires_grad=True)

        # init bias
        if bias:
            self.bias = torch.randn((self.out_channels, ), requires_grad=True)
    
    # calculate conv3d use kernel Y on X
    @staticmethod
    def conv4d(X: torch.Tensor, Y: torch.Tensor, output: torch.Tensor):
        dim1_cal = X.shape[0] - Y.shape[0] + 1
        dim2_cal = X.shape[1] - Y.shape[1] + 1
        dim3_cal = X.shape[2] - Y.shape[2] + 1
        dim4_cal = X.shape[3] - Y.shape[3] + 1
        for i_1 in range(dim1_cal):
            for i_2 in range(dim2_cal):
                for i_3 in range(dim3_cal):
                    for i_4 in range(dim4_cal):
                        output[i_1][i_2][i_3][i_4] += torch.sum(X[i_1:i_1 + Y.shape[0], \
                                                            i_2:i_2 + Y.shape[1], \
                                                            i_3:i_3 + Y.shape[2], \
                                                            i_4:i_4 + Y.shape[3]] * Y)

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[0] == self.in_channels, 'in channels not match!'
        assert input.shape[1] >= self.kernel_size[0] and \
               input.shape[2] >= self.kernel_size[1] and \
               input.shape[3] >= self.kernel_size[2] and \
               input.shape[4] >= self.kernel_size[3], 'input to less!'
        calculated = torch.zeros(self.out_channels, input.shape[1] - self.kernel_size[0] + 1, \
                                                    input.shape[2] - self.kernel_size[1] + 1, \
                                                    input.shape[3] - self.kernel_size[2] + 1, \
                                                    input.shape[4] - self.kernel_size[3] + 1)
        for i_out in range(self.out_channels):
            for i_in in range(self.in_channels):
                self.conv4d(input[i_in], self.parameters[i_out, i_in], calculated[i_out])
        if self.use_bias:
            calculated += self.bias.reshape((self.out_channels, 1, 1, 1, 1))
        return calculated

    def zero_grad(self):
        self.parameters.grad.zero_()
        if self.use_bias:
            self.bias.grad.zero_()
