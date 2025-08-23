
import torch
import numpy as np
import torch.nn.functional as F
import math


class LinearDummy(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels))
        self.bias=None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, gain=1):
        return F.linear(x, self.weight, self.bias)


class BlockSiren(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True, is_first_layer=False, scale_init=1.0):
        super(BlockSiren, self).__init__()
        self.use_bias = use_bias
        # self.activ = activ
        self.is_first_layer = is_first_layer
        self.scale_init = scale_init
        # self.freq_scaling = scale_init

        self.conv = torch.nn.Linear(in_channels, out_channels, bias=self.use_bias).cuda()

        with torch.no_grad():
            
            #following the official implementation from https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb
            if self.is_first_layer:
                self.conv.weight.uniform_(-1 / in_channels, 
                                            1 / in_channels)      
            else:
                self.conv.weight.uniform_(-np.sqrt(6 / in_channels) / self.scale_init, 
                                            np.sqrt(6 / in_channels) / self.scale_init)
                
            self.conv.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        
        x = self.scale_init * x
        
        x = torch.sin(x)

        return x


class LinearWN_v2(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWN_v2, self).__init__(in_features, out_features, bias)
        self.g = torch.nn.Parameter(torch.ones(out_features))
        

        self.in_features=in_features
        self.out_features=out_features

    def forward(self, input):
        w= torch._weight_norm(self.weight, self.g, 0)
        out=F.linear(input, w, self.bias)
        return out
    
    

class Conv1dWN_v2(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        padding_mode="zeros",
        dilation=1,
        groups=1,
    ):
        super(Conv1dWN_v2, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            True,
        )
        self.g = torch.nn.Parameter(torch.ones(out_channels))
        
        self.padding_amount=padding
        self.padding_mode=padding_mode

    def forward(self, x):
        w= torch._weight_norm(self.weight, self.g, 0)



        if self.padding_mode != 'zeros':
            x= F.pad(x, (self.padding_amount,self.padding_amount), mode=self.padding_mode)
        
            x= F.conv1d(
                    x,
                    w,
                    bias=self.bias,
                    stride=self.stride,
                    padding=0,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            # print("x after conv", x.shape)
            return x
        else:
        
            return F.conv1d(
                x,
                w,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )


def kaiming_init(m, is_linear, nonlinearity="silu"):
    # gain = math.sqrt(2.0 / (1.0 + alpha**2))

    # gain=np.sqrt(10.5)
    if nonlinearity=="silu":
        gain=np.sqrt(2.3) #works fine with silu
    elif nonlinearity=="relu":
        gain=np.sqrt(2) #works fine with silu
    # gain=np.sqrt(2.15)
    # gain=np.sqrt(0.92) #for mpsilu
    scale=1.0

    if is_linear:
        gain = 1

    # print("effective scale", gain*scale)
        
    # print("m is ",m)
    # help(m)

    if isinstance(m, torch.nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        # std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
        std = gain * math.sqrt(n1 * ksize)
    elif isinstance(m, torch.nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        # std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
        std = gain * math.sqrt(n1 * ksize)
    elif isinstance(m, torch.nn.Conv1d):
        ksize = m.kernel_size[0] 
        n1 = m.in_channels
        n2 = m.out_channels
        # std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
        std = gain * math.sqrt(n1 * ksize)
    elif isinstance(m, torch.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        # std = gain * math.sqrt(2.0 / (n1 + n2))
        std = gain * math.sqrt(n1)
    # elif isinstance(m, SMConv1d_v2):
    #     print("SMConv1d_v2")
    #     exit()
    else:
        return
    
    # help(m)
    
    # print("m is ", m)
    
    # print("std is", std)

    # m.weight.data.normal_(0, std)
    # m.weight.data.uniform_(-std * scale, std * scale)
    fan = torch.nn.init._calculate_correct_fan(m.weight, "fan_in")
    std = gain / math.sqrt(fan)
    # print("std is", std)
    with torch.no_grad():
        # m.weight.normal_(0, std)
        m.weight.data.uniform_(-std * math.sqrt(3.0) * scale, std * math.sqrt(3.0) * scale)
    if m.bias is not None:
        m.bias.data.zero_()

    if isinstance(m, torch.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    # if isinstance(m, Conv2dWNUB) or isinstance(m, ConvTranspose2dWNUB) or isinstance(m, LinearWN):
    # print("m is ", m)
    if (
        # isinstance(m, torch.Conv2dWNUB)
        # isinstance(m, torch.Conv2dWN)
        # or isinstance(m, torch.ConvTranspose2dWN)
        # or isinstance(m, torch.ConvTranspose2dWNUB)
        isinstance(m, LinearWN_v2)
        or isinstance(m, Conv1dWN_v2)
    ):
        # print("selected m is ", m)
        # help(m)
        # norm = np.sqrt(torch.sum(m.weight.data[:] ** 2))
        dims = list(range(1, len(m.weight.shape)))
        norm = torch.norm(m.weight, 2, dim=dims, keepdim=True)
        # print("weight is ", m.weight.shape)
        # print("norm",norm.shape)
        # print("m.g.data",m.g.data.shape)
        # norm = torch.norm(m.weight, 2, dim=0, keepdim=True)
        m.g.data = norm

    