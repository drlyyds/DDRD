import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs):
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x


class ConvBlock(nn.Module):
    """ Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, dropout, activation=nn.ReLU()):
        super(ConvBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            ConvNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="tanh",
            ),
            nn.BatchNorm1d(out_channels),
            activation
        )
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, enc_input, mask=None):
        enc_output = enc_input.contiguous().transpose(1, 2)
        enc_output = F.dropout(self.conv_layer(enc_output), self.dropout, self.training)

        enc_output = self.layer_norm(enc_output.contiguous().transpose(1, 2))
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output


class ConvNorm(nn.Module):
    """如果传入参数为transpose=False，那么输入形状为 (B, 输入通道数, seq_len)， 输出形状为 (B, 输出通道数, seq_len)，在实例化对象的时候，传入的第一个参数是输入通道数，第二个参数为输出通道数\n
       如果传入参数为transpose=true，那么输入形状为 (B, seq_len, 输入通道数)， 输出形状为 (B, 输出通道数, seq_len)，在实例化对象的时候，传入的第一个参数是输入通道数，第二个参数为输出通道数"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1, # 卷积核的膨胀率，默认为 1
        bias=True,  # 是否使用偏置项，默认为 True
        w_init_gain="linear",  # 权重初始化的增益类型，默认为 "linear"
        transpose=False,   # 是否进行维度转置，默认为 False
    ):
        # 调用父类的构造函数
        super(ConvNorm, self).__init__()

        # 如果 padding 为 None，则根据卷积核大小和膨胀率自动计算 padding
        if padding is None:
            # 确保卷积核大小为奇数，以便正确计算 padding
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        # 定义一维卷积层
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        # 使用 Xavier 均匀分布初始化卷积层的权重
        # 根据 w_init_gain 计算增益值，用于调整初始化的分布范围
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        # 记录是否需要进行维度转置
        self.transpose = transpose

    def forward(self, x):
        # 如果需要进行维度转置
        if self.transpose:
            # 对输入进行维度转置，将 (batch_size, in_channels, seq_len) 转换为 (batch_size, seq_len, in_channels)
            x = x.contiguous().transpose(1, 2)
        # 对输入进行卷积操作
        x = self.conv(x)
        # 如果之前进行了维度转置，则再转置回来
        if self.transpose:
            x = x.contiguous().transpose(1, 2)

        return x
