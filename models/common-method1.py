# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import logging
import math
import warnings
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils.datasets import exif_transpose, letterbox
from utils.general import colorstr, increment_path, is_ascii, make_divisible, non_max_suppression, save_one_box, \
    scale_coords, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync

import torch.nn.functional as F
from torch.nn.parameter import Parameter
import copy
import pandas as pd
import math
import os
LOGGER = logging.getLogger(__name__)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

#Conv(c_, c2, 3, 1, g=g)
class Conv_pt(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_pt, self).__init__()
        self.conv_pt = SubConv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        #self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        #self.bn = nn.BatchNorm2d(c2)
        self.bn = BatchNorm(c2) 
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv_pt(x)))

    def fuseforward(self, x):
        return self.act(self.conv_pt(x))


class SubConv2d(nn.Conv2d):
    # Standard convolution        Conv(c1 * 4, c2, k, s, p, g, act)
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, bias=False):  # ch_in, ch_out, kernel, stride, padding, groups 
        #执行和初始化父类nn.Conv2d的构造方法，def __init__() 是子类自己的初始化构造方法  in_channels代表channel数，out_channels代表filters的数量
        super(SubConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding=padding, groups=groups, bias=bias)
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, p), groups=g, bias=False)
        #self.bn = nn.BatchNorm2d(c2)

        # self.bn = BatchNorm(out_channels)
        #self.act = nn.Hardswish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        #super(FilterStripe, self).__init__(in_channels, out_channels, kernel_size, stride, kernel_size // 2, groups=1, bias=False)

        self.BrokenTarget = None
        # 通过Parameter来定义 filter 的 skeleton, 即filter的channel数，长和宽
        # 原项目的 FS初始化值是1, 而yolo self.weight的初始化值是一个空tensor，由torch.empty产生
        # 这里把filter的数量(out_channels) 设为了channel数，为的就是在剪枝的时候一个channel用来裁剪一个filter 
        self.FilterSkeleton = Parameter(torch.ones(self.out_channels, self.kernel_size[0], self.kernel_size[1]), requires_grad=True)


    # def forward(self, x):
    #     return self.act(self.bn(self.conv(x)))
    
    def forward(self, x):
        # 在prune之前,BT都是none, 因此都是走else的,只有在prune之后,才走if 里的.
        if self.BrokenTarget is not None:
            # 维度顺序依次是：x.shape[0]--filter数量，channel 数量，卷积核尺寸-长，卷积核尺寸-宽）；   numpy.ceil 向正无穷取整
            # >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
            # >>> np.ceil(a)
            # array([-1., -1., -0.,  1.,  2.,  2.,  2.])
            # out = torch.zeros(x.shape[0], self.FilterSkeleton.shape[0], int(np.ceil(x.shape[2] / self.stride[0])), int(np.ceil(x.shape[3] / self.stride[1])))
            # if x.is_cuda:
            #     out = out.cuda()
            # x = F.conv2d(x, self.weight) #x.float()
            #
            # l, h = 0, 0
            # for i in range(self.BrokenTarget.shape[0]):
            #     for j in range(self.BrokenTarget.shape[1]):
            #         h += self.FilterSkeleton[:, i, j].sum().item()
            #         out[:, self.FilterSkeleton[:, i, j]] += self.shift(x[:, l:h], i, j)[:, :, ::self.stride[0], ::self.stride[1]]
            #         l += self.FilterSkeleton[:, i, j].sum().item()
            # return out
            return F.conv2d(x, self.weight)
        else:
            ###self.conv 只有在forward中运行了，才会被纳入graph中，才会在loss.backward时计算_grad; 因此这里加入* self.FilterSkeleton.unsqueeze(1)目的是想要让FS的数据在loss时被grad
            #https://blog.csdn.net/Bear_Kai/article/details/102698364
            # 原始方法
            # weight_test = self.weight.clone()
            # FS_test = self.FilterSkeleton.clone()

            # x1 = F.conv2d(x.float(), self.weight, stride=self.stride, padding=self.padding, groups=self.groups)
            # y = F.conv2d(x.float(), self.weight * self.FilterSkeleton.unsqueeze(1), stride=self.stride, padding=self.padding, groups=self.groups)
            # out = x1 + y * 0
            # print('self.weight', self.weight.shape)
            # print('self.FilterSkeleton', self.FilterSkeleton.shape)
            return F.conv2d(x, self.weight * self.FilterSkeleton.unsqueeze(1), stride=self.stride, padding=self.padding, groups=self.groups)
            # return F.conv2d(x.float(), self.weight, stride=self.stride, padding=self.padding, groups=self.groups)
            # return out
            # 不能用以下方法，因为当self.FilterSkeleton 下降到0时，就会出现与self.weight相乘为0的情况。
            # return F.conv2d(x.float(), self.weight * (self.FilterSkeleton.unsqueeze(1) * (1 / self.FilterSkeleton.unsqueeze(1))), stride=self.stride, padding=self.padding, groups=self.groups)
            # return F.conv2d(x.float(), self.weight + (self.FilterSkeleton.unsqueeze(1) * 0).sum(), stride=self.stride, padding=self.padding, groups=self.groups)


    # def prune_in(self, in_mask=None):
    #     self.weight = Parameter(self.weight[:, in_mask])
    #     self.in_channels = in_mask.sum().item()

    # def prune_out(self, threshold):
    #     out_mask = (self.FilterSkeleton.abs() > threshold).sum(dim=(1, 2)) != 0
    #     if out_mask.sum() == 0:
    #         out_mask[0] = True
    #     # print('shape-mask---', out_mask.shape)
    #     # print('self.weight', self.weight.shape)
    #     self.weight = Parameter(self.weight[out_mask])
    #     self.FilterSkeleton = Parameter(self.FilterSkeleton[out_mask], requires_grad=True)
    #     self.out_channels = out_mask.sum().item()
       
    #     return out_mask

    def prune_in(self, in_mask=None):
        self.weight = Parameter(self.weight[:, in_mask])
        self.in_channels = in_mask.sum().item()

    def prune_out(self, threshold):
        # sum(dim=(1, 2)) 同一filter内所有channel相加，形成一个channel, 然后在该channel内所有行纵向相加，只剩一行; 剩下的是（filters, channel(1), row(1), coloum(3）)
        # print('self.weight-01', self.weight.shape)
        out_mask = (self.FilterSkeleton.abs() > threshold).sum(dim=(1, 2)) != 0

        self.weight = Parameter(self.weight[out_mask])
        # print('self.weight-02', self.weight.shape)
        self.FilterSkeleton = Parameter(self.FilterSkeleton[out_mask], requires_grad=True)
        self.out_channels = out_mask.sum().item()
        return out_mask

    def _break(self, threshold):
        # 这里的weight和FS相乘,是对应了论文中FS 和 weight的点乘, 因为这两个在更新时都是单独更新的, 所以论文里需要他们进行点乘, 这一步是在这里进行了实现. 
        self.weight = Parameter(self.weight * self.FilterSkeleton.unsqueeze(1)) #a.squeeze(N) 就是在a中指定位置N加上一个维数为1的维度。。a.squeeze(N) 就是在a中指定位置N加上一个维数为1的维度。
        self.FilterSkeleton = Parameter((self.FilterSkeleton.abs() > threshold), requires_grad=False)
        #如果没有值小于threshold, 即不需要剪枝,则把三个维度的所有值都设为True(这个功能优待测试)
        if self.FilterSkeleton.sum() == 0:
            self.FilterSkeleton.data[0][0][0] = True

        # 这里如果没有剪枝,就不给BrokenTarget 赋值,从而不用在forward的if BrokenTarget 里跑,这样就尽可能提高准确率

        self.out_channels = self.FilterSkeleton.sum().item()

        self.BrokenTarget = self.FilterSkeleton.sum(dim=0)

        self.kernel_size = (1, 1)
        
        self.weight = Parameter(self.weight.permute(2, 3, 0, 1).reshape(-1, self.in_channels, 1, 1)[self.FilterSkeleton.permute(1, 2, 0).reshape(-1)])

    def update_skeleton(self, sr, threshold, key):
        FS_grad = copy.deepcopy(self.FilterSkeleton.grad.data)
        ## 通过传入的sr更新FS 梯度, 就是论文中公式5的后面部分或者公式6; g(I)相当于是FS的grad.data; 因为FS也在Optimizer中更新，所以
        # 这里只需要在fs的grad上进行更改即可; 公式6整体的L只是用来计算FS的
        # 符号函数，返回一个新张量，包含输入input张量每个元素的正负（大于0的元素对应1，小于0的元素对应-1，0还是0, 这里FS只有可能是1或者0, 乘以sr后得到一个Musk
        
        self.FilterSkeleton.grad.data.add_(sr * torch.sign(self.FilterSkeleton.data))   # add_()对原始的tensor的每一个数值进行加的操作
        # 通过FS的data和threshold来设置mask
        mask = self.FilterSkeleton.data.abs() > threshold 

        ### 通过与mask(0/1矩阵)相乘来得到FS形状
        self.FilterSkeleton.data.mul_(mask)   # torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等; 这里是FS的data和 mask 对应位置相乘
        #print('self.FilterSkeleton.data.mul_(mask)=====', self.FilterSkeleton.data)
        self.FilterSkeleton.grad.data.mul_(mask)        # 这里是FS的gradient和 mask 对应位置相乘
        # remove kernel size dimention, only channel left.
        # 把所有FS中为0的stripe消掉
        out_mask = mask.sum(dim=(1, 2)) != 0
        return out_mask, FS_grad

    def Elastic_Net_penalty(self, sr, threshold): #delta 
        FS_grad = copy.deepcopy(self.FilterSkeleton.grad.data)
        # 原公式
        #self.FilterSkeleton.grad.data.add_(sr * torch.sign(self.FilterSkeleton.data))   # add_()对原始的tensor的每一个数值进行加的操作
        # Method02 公式
        self.FilterSkeleton.grad.data.add_(torch.sign(self.FilterSkeleton.data) * (2 * sr * self.FilterSkeleton.data.abs() + (1 - sr)))
        # 通过FS的data和threshold来设置mask
        mask = self.FilterSkeleton.data.abs() > threshold 

        ### 通过与mask(0/1矩阵)相乘来得到FS形状
        self.FilterSkeleton.data.mul_(mask)   

        self.FilterSkeleton.grad.data.mul_(mask)        # 这里是FS的gradient和 mask 对应位置相乘

        out_mask = mask.sum(dim=(1, 2)) != 0

        return out_mask, FS_grad

    # 跟踪某几个FS中的weight在训练过程中的变化并输出
    # indexes: 所有候选FS weight, key: 层名字 i: batch序号; flag: 判断要打印的是sensitive strip还是insensitive的；sensitive 就是那些删了后准确率掉的大的
    def follow_stripe_update(self, indexes, key, epoch, i, flag, FS_grad, sr, threshold, deriv=None):    # FS_01, 

        orig_path = './runs/stripe_experiment/' + str(sr)+ '_'+ str(threshold) + '/'
        if os.path.exists(orig_path) is False:
            os.mkdir(orig_path)

        results, results_1st, results_2nd = [], [], []
        results_FS_01, results_FS_grad = [], []
        # 将所有候选的weight的坐标列出, 放入result
        for idx in indexes:
            #必须将FS转化为Numpy, 才能运用使用numpy获取的坐标.
            FS = copy.deepcopy(self.FilterSkeleton.data)
            FS = FS.cpu().detach().numpy()
            results.append(FS[idx[0], idx[1], idx[2]]) # 必须是数字的形式, 而不是append([a, b ,c]), 是 append(a, b, c)

            FS_grad = FS_grad.detach()
            results_FS_grad.append(FS_grad[idx[0], idx[1], idx[2]]) # 必须是数字的形式, 而不是append([a, b ,c]), 是 append(a, b, c)


        #初始时设置标题, 往后就不设置标题了
        if epoch == 0 and i == 0:  # 当epoch=0 , batch =1, 设置标题
            indexes = [str(x) for x in indexes]
            file =pd.DataFrame([results], index= [epoch], columns=indexes)      # index设置行号, 这里设为epoch, columns设标题, 这里将所有index设置为标题 
            
            if flag == 'sensitive':
                
                file.to_csv(orig_path + 'follow_update_new_sensitive_' +key +".csv", mode='a+')  
            else:
                file.to_csv(orig_path + 'follow_update_new_insensitive_' +key +".csv", mode='a+') 
        else:
            file =pd.DataFrame([results], index= [epoch], columns=None)
            
            if flag == 'sensitive':
                file.to_csv(orig_path + 'follow_update_new_sensitive_' +key +".csv", mode='a+', header=False) 
            else:
                file.to_csv(orig_path + 'follow_update_new_insensitive_' +key +".csv", mode='a+', header=False) 
        results.clear()  # 清空列表
    

        #初始时设置标题, 往后就不设置标题了
        if epoch == 0 and i == 0:  # 当epoch=0 , batch =1, 设置标题
            indexes = [str(x) for x in indexes]
            file =pd.DataFrame([results_FS_grad], index= [epoch], columns=indexes)      # index设置行号, 这里设为epoch, columns设标题, 这里将所有index设置为标题 
            
            if flag == 'sensitive':
                file.to_csv(orig_path +  'follow_update_new_sensitive_indexes_FS_grad' +key +".csv", mode='a+')  
            else:
                file.to_csv(orig_path +  'follow_update_new_insensitive_indexes_FS_grad' +key +".csv", mode='a+') 
        else:
            file =pd.DataFrame([results_FS_grad], index= [epoch], columns=None)
            
            if flag == 'sensitive':
                file.to_csv(orig_path + 'follow_update_new_sensitive_indexes_FS_grad' +key +".csv", mode='a+', header=False) 
            else:
                file.to_csv(orig_path + 'follow_update_new_insensitive_indexes_FS_grad' +key +".csv", mode='a+', header=False) 
        results_FS_grad.clear()  # 清空列表

    def shift(self, x, i, j):
        return F.pad(x, (self.BrokenTarget.shape[0] // 2 - j, j - self.BrokenTarget.shape[0] // 2, self.BrokenTarget.shape[0] // 2 - i, i - self.BrokenTarget.shape[1] // 2), 'constant', 0)

    def extra_repr(self):
        s = ('{BrokenTarget},{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        return s.format(**self.__dict__)

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm, self).__init__(num_features)
        self.weight.data.fill_(0.5)
    def prune(self, mask=None):
        #print('bn-01============', Parameter(self.weight).shape)
        #print('mask---', mask)
        self.weight = Parameter(self.weight[mask])
        
        self.bias = Parameter(self.bias[mask])
        #print('bias', self.bias.shape)
        self.register_buffer('running_mean', self.running_mean[mask])
        self.register_buffer('running_var', self.running_var[mask])
        self.num_features = mask.sum().item()

    def update_mask(self, mask=None, threshold=None):
        if mask is None:
            mask = self.weight.data.abs() > threshold
        self.weight.data.mul_(mask)
        self.bias.data.mul_(mask)
        self.weight.grad.data.mul_(mask)
        self.bias.grad.data.mul_(mask)


# 加入mobilenet结构--------------------------------------------------------------------------------------------------------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            # nn.ReLU6(inplace=True)
            nn.SiLU(inplace=True)
        )

    def fuseforward(self, input):
        for module in self:
            if module is None or type(module) is nn.BatchNorm2d:
                continue
            input = module(input)
        return input


class ConvBNReLU_Stripe(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU_Stripe, self).__init__(
            # nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            SubConv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            # nn.BatchNorm2d(out_planes),
            BatchNorm(out_planes),
            #nn.ReLU6(inplace=True)
            nn.SiLU(inplace=True)

        )

    def fuseforward(self, input):
        for module in self:
            if module is None or type(module) is BatchNorm:
                continue
            input = module(input)
        return input

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU_Stripe(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            #ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def fuseforward(self, x):
        for i, module in enumerate(self.conv):
            if module is None or type(module) is nn.BatchNorm2d:
                del self.conv[i]
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class BottleneckMOB(nn.Module):
    # c1:inp  c2:oup s:stride  expand_ratio:t
    def __init__(self, c1, c2, s, expand_ratio):
        super(BottleneckMOB, self).__init__()
        self.s = s
        hidden_dim = round(c1 * expand_ratio)
        self.use_res_connect = self.s == 1 and c1 == c2
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, s, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, c2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c2),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(c1, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, s, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, c2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c2),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
# ---------------------------------------------------------------------------------------------------------------------------


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        #self.cv2 = Conv_pt(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))



class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[GhostBottleneck(c_, c_) for _ in range(n)])


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        #---------------------------------------------------------------------------------------------------------------------------------------
        #self.conv = Conv_pt(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class
    max_det = 1000  # maximum number of detections per image

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def autoshape(self):
        LOGGER.info('AutoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, pil=not self.ascii)
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                str += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        LOGGER.info(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
