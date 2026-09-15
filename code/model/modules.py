import os
import json
import copy
import math
from collections import OrderedDict
from transformers import ClapModel, ClapProcessor
from torch.nn import ModuleList, LayerNorm, Linear
import tgt
import torch
import torch.nn as nn
from numba import jit, prange
import numpy as np
import torch.nn.functional as F
import librosa
from utils.tools import (
    get_variance_level,
    get_phoneme_level_pitch,
    get_phoneme_level_energy,
    get_mask_from_lengths,
    pad_1D,
    pad,
)
from text.symbols import symbols
from .transformers.transformer import MultiHeadAttention, PositionwiseFeedForward
from .transformers.constants import PAD
from .transformers.blocks import get_sinusoid_encoding_table, Swish, LinearNorm, ConvNorm, ConvBlock
from transformers import AutoTokenizer, AutoModel
from transformers import BertModel, Wav2Vec2Model,DistilBertModel
from transformers import  BertModel
from sentence_transformers import SentenceTransformer


rank = 0
device = torch.device('cuda:{}'.format(rank) if torch.cuda.is_available() else 'cpu')
@jit(nopython=True)
def mas_width1(attn_map):
    """
    输入：attn_map 是一个二维的注意力矩阵，其形状通常为 (mel_frames, text_length)也就是(实际音频帧数，实际音素序列个数)，其中 mel_frames 表示梅尔频谱的帧数，text_length 表示音素序列的长度。矩阵中的每个元素表示对应音频帧与音素之间的注意力得分（通常是经过 softmax 处理后的概率值）。\n
    输出：opt 是一个与 attn_map 形状相同的二进制矩阵(实际音频帧数，实际音素序列个数)，矩阵中值为 1 的位置表示该位置被选中作为最优路径的一部分，值为 0 的位置表示未被选中。\n
    这个 mas_width1算法是在 “mel‐frames × text‐steps” 矩阵里找一条从左上到右下的单调路径，所以：每一行（对应一个 帧）恰好选一个 音素索引，opt[i, :] 上只有一个 1。而每一列（对应一个音素）可能会被多帧重复选中，但行上一定是一对一。"""
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]): # for each text dim
            prev_log = log_p[i - 1, j]
            prev_j = j

            if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1

            log_p[i, j] = attn_map[i, j] + prev_log
            prev_ind[i, j] = prev_j

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


@jit(nopython=True, parallel=True)
def b_mas(b_attn_map, in_lens, out_lens, width=1):
    """b_mas 函数用于对批量的注意力矩阵进行处理，针对每个注意力矩阵找出满足宽度为 1 约束的最优对齐路径，并把这些路径信息存于与输入形状相同的矩阵中。\n
    输入参数:
        b_attn_map：一个四维的批量注意力矩阵，形状一般是(B,1,D,T)----- (batch_size, 1, max_out_length, max_in_length)，max_out_length 表示批量中所有样本的帧数的最大值，max_in_length 表示批量中所有样本的音素序列长度的最大值。\n
        in_lens：一维数组，长度为 batch_size，存储每个音频的实际音素序列长度。\n
        out_lens：一维数组，长度为 batch_size，存储每个音频的实际帧长。\n
        width：对齐路径的宽度约束，该函数要求此值必须为 1。
    return:
        返回一个与输入 b_attn_map 形状相同的矩阵 attn_out，其中存储了有效的batch中每个音频的音素与帧的对齐路径信息(用0，1来表示)并且包含无效的0部分。
    """
    assert width == 1
    attn_out = np.zeros_like(b_attn_map)

    for b in prange(b_attn_map.shape[0]): #prange可以理解为range，只是开启了并行计算
        # b_attn_map[b, 0, : out_lens[b], : in_lens[b]]，将第b个音频的注意力矩阵(D,T)裁剪成有效注意力矩阵(实际帧长，实际音素序列长度)，然后将这个有效注意力矩阵送入mas_width1函数
        out = mas_width1(b_attn_map[b, 0, : out_lens[b], : in_lens[b]])  #out为有效的----帧与音素序列对应矩阵(用0，1表示)，shape为(实际音频帧数，实际音素序列个数)
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out   #将out赋值给attn_out
    return attn_out


class PostNet(nn.Module):
    """
    后处理网络：由五个一维卷积层组成，每个卷积层有 512 个通道，卷积核大小为 5 \n
    输入为(B, 音频帧数, 80)，输出也为(B, 音频帧数, 80)
    """

    def __init__(
        self,
        n_mel_channels=80, # 输入的梅尔频谱通道数，默认为 80
        postnet_embedding_dim=512,  # 卷积层的嵌入维度，即通道数，默认为 512
        postnet_kernel_size=5,  # 卷积核的大小，默认为 5
        postnet_n_convolutions=5, # 卷积层的数量，默认为 5
    ):

        super(PostNet, self).__init__()
        # 创建一个 ModuleList 来存储卷积层序列
        self.convolutions = nn.ModuleList()

        # 添加第一个卷积层序列
        self.convolutions.append(
            nn.Sequential(
                # 自定义的一维卷积层，输入通道数为 n_mel_channels，输出通道数为 postnet_embedding_dim
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2), # 填充大小，保证输入输出长度相同(尺寸，长和高？)
                    dilation=1,
                    w_init_gain="tanh",  # 权重初始化的增益类型为 tanh
                ),
                # 一维批量归一化层，对卷积层的输出进行归一化处理
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        # 添加中间的卷积层序列（除了第一个和最后一个）,range(start, stop) 在 Python 中是左闭右开（包含 start，不包含 stop）。
        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        # 添加最后一个卷积层序列
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )

    def forward(self, x):
        # 将输入的维度进行转置，从 (batch_size, seq_len, n_mel_channels) 转换为 (batch_size, n_mel_channels, seq_len)
        x = x.contiguous().transpose(1, 2)

        # 遍历除最后一个卷积层序列之外的所有卷积层序列
        for i in range(len(self.convolutions) - 1):
            # 对当前卷积层序列的输出应用 tanh 激活函数,应用 dropout 正则化，丢弃率为 0.5
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)

        # 对最后一个卷积层序列的输出应用 dropout 正则化
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        x = x.contiguous().transpose(1, 2)
        return x

"""Variance Predictor（方差预测器）：是一个具体的子网络，专门用于预测某一个韵律特征（例如单独预测时长、音高或能量）。每个 Predictor 接收输入特征，然后输出对应属性的预测值。
Variance Adaptor（方差适配器）：是一个更高层次的模块，它整合了多个 Variance Predictor。除了调用各个 Predictor 来获得预测值之外，
            还负责利用这些预测结果对输入的文本嵌入进行调整（例如通过扩展时长、加入 pitch 或能量 embedding），使得最终的特征能更好地指导语音生成。"""


"""方差预测器这个名字来源于 FastSpeech 2，因为这些韵律特征（时长、音高、能量）在不同语境下会有较大的变化，被认为是语音合成中的方差信息（variance information），
因此相关预测器被称为“方差预测器”。但本质上，它们输出的就是这些特征的具体数值，而不是统计上的方差。"""
class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, preprocess_config, model_config, train_config):
        super(VarianceAdaptor, self).__init__()
        #实例化时长预测器，预测文本每个音素对应的持续时间，如果想使用专门的时长预测器，也可以使用 DurationPredictor（目前注释掉）
        self.duration_predictor = VariancePredictor(model_config)
        # self.duration_predictor = DurationPredictor(model_config)

        #实例化长度调节器，用于根据预测的时长扩展文本嵌入到与 mel 长度一致
        self.length_regulator = LengthRegulator()

        # 实例化音高和能量预测器，结构与时长预测器类似（都是 VariancePredictor），虽然类一样，但是这是两个对象，也就是模型的参数是不共享的，即使输入相同结果也不会相同，只是形状一样罢了
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        # 从模型配置中获取是否采用学习对齐机制,另外yaml配置文件如果是键值对返回的是字典
        self.learn_alignment = model_config["duration_modeling"]["learn_alignment"]    #其实self.learn_alignment=true
        # 从训练配置中获取开始进行 binarization 的步数
        self.binarization_start_steps = train_config["duration"]["binarization_start_steps"]   #6000

        # 如果采用学习对齐，则构造对齐器，用于 unsupervised 时长建模
        if model_config["duration_modeling"]["learn_alignment"]:   #是true
            self.aligner = AlignmentEncoder(
                n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],    #80
                n_att_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],    #80
                n_text_channels=model_config["transformer"]["encoder_hidden"],   #256
                temperature=model_config["duration_modeling"]["aligner_temperature"],   #0.0005
                multi_speaker=model_config["multi_speaker"],   #true
                multi_emotion=model_config["multi_emotion"],   #true
            )

        # 四个变量的结果是 "frame"，"frame","phone","phone"
        pitch_level_tag, energy_level_tag, self.pitch_feature_level, self.energy_feature_level = \
                                    get_variance_level(preprocess_config, model_config, data_loading=False)

        # Note that there is no pre-extracted phoneme-level variance features in unsupervised duration modeling.
        # Alternatively, we can use convolutional embedding instead of bucket-based embedding in such cases.
        self.use_conv_embedding = self.learn_alignment \
            and (self.pitch_feature_level == "phoneme_level" or self.energy_feature_level == "phoneme_level")
        """use_conv_embedding=true"""
        if self.use_conv_embedding:
            kernel_size = model_config["variance_embedding"]["kernel_size"]  #9
            self.pitch_embedding = ConvNorm(
                1,
                model_config["transformer"]["encoder_hidden"],   #256
                kernel_size=kernel_size,  # 9
                stride=1,
                padding=int((kernel_size - 1) / 2),   #4
                bias=False,
                w_init_gain="tanh",
                transpose=True,
            )
            self.energy_embedding = ConvNorm(
                1,
                model_config["transformer"]["encoder_hidden"],
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                bias=False,
                w_init_gain="tanh",
                transpose=True,
            )
        else:
            pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
            energy_quantization = model_config["variance_embedding"]["energy_quantization"]
            n_bins = model_config["variance_embedding"]["n_bins"]
            assert pitch_quantization in ["linear", "log"]
            assert energy_quantization in ["linear", "log"]
            with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
            ) as f:
                stats = json.load(f)
                pitch_min, pitch_max = stats[f"pitch_{pitch_level_tag}"][:2]
                energy_min, energy_max = stats[f"energy_{energy_level_tag}"][:2]

            if pitch_quantization == "log":
                self.pitch_bins = nn.Parameter(
                    torch.exp(
                        torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                    ),
                    requires_grad=False,
                )
            else:
                self.pitch_bins = nn.Parameter(
                    torch.linspace(pitch_min, pitch_max, n_bins - 1),
                    requires_grad=False,
                )
            if energy_quantization == "log":
                self.energy_bins = nn.Parameter(
                    torch.exp(
                        torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                    ),
                    requires_grad=False,
                )
            else:
                self.energy_bins = nn.Parameter(
                    torch.linspace(energy_min, energy_max, n_bins - 1),
                    requires_grad=False,
                )

            self.pitch_embedding = nn.Embedding(
                n_bins, model_config["transformer"]["encoder_hidden"]
            )
            self.energy_embedding = nn.Embedding(
                n_bins, model_config["transformer"]["encoder_hidden"]
            )

    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """输入参数为 1.attn：来自AlignmentEncoder的注意力矩阵形状为(B,1,D,T);
        2.in_lens：一维数组，长度为 batch_size，存储每个音频的实际音素序列长度;3.out_lens：一维数组，长度为 batch_size，存储每个音频的实际帧长。\n

        return为：返回一个与输入形状相同的tensor attn_out(B,1,D,T)，在GPU上，并且其中存储了batch中每个音频的帧音素的对齐路径信息(用0，1来表示) 并且包含padding的全0部分"""

        #没有梯度计算，并且在CPU上进行计算
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)
        return torch.from_numpy(attn_out).to(attn.device)

    def get_phoneme_level_pitch(self, duration, src_len, pitch_frame):
        """输入duration, src_len, pitch_frame。duration → NumPy 数组，形状 (B, T)，每个元素是某条音频样本某个音素的预测帧长； src_len→ (B,)，表示一个batch中每条音频样本实际有多少个音素（去掉 padding）；
           pitch_frame → (B, D)， 每帧的pitch 数值。D为batch中最大帧数 \n
           输出音素级pitch为二维numpy数组，shape为(B,T)T为batch中音频的最大音素序列长度，每个元素为对应音素的音素级pitch就是(音素对应的) 插值过的帧级pitch取平均，放在GPU上"""
        return torch.from_numpy(
            pad_1D(
                [get_phoneme_level_pitch(dur[:len], var) for dur, len, var \
                        in zip(duration.int().cpu().numpy(), src_len.cpu().numpy(), pitch_frame.cpu().numpy())]
            )
        ).float().to(pitch_frame.device)

    def get_phoneme_level_energy(self, duration, src_len, energy_frame):
        """输入duration, src_len, energy_frame。duration → NumPy 数组，形状 (B, T)，每个元素是某条音频样本某个音素的预测帧长； src_len→ (B,)，表示一个batch中每条音频样本实际有多少个音素（去掉 padding）；
           energy_frame → (B, D)， 每帧的energy值。D为batch中最大帧数 \n
           输出音素级energy 为二维numpy数组，shape为(B,T)T为batch中音频的最大音素序列长度，每个元素为对应音素的音素级energy就是(音素对应的)的帧级energy取平均，放在GPU上"""
        return torch.from_numpy(
            pad_1D(
                [get_phoneme_level_energy(dur[:len], var) for dur, len, var \
                        in zip(duration.int().cpu().numpy(), src_len.cpu().numpy(), energy_frame.cpu().numpy())]
            )
        ).float().to(energy_frame.device)


    def get_pitch_embedding(self, x, target, mask, control):
        """传入参数为x, target, mask, control， 其中mask的形状为(Batchsize,T)；x这个参数来自于transformers.py中的TextEncoder的输出结果，形状为(Batchsize, T，256)，T为当前batch中音素序列的最大长度，(因为不满足的在dataset中会被扩充)
        target训练阶段不是None，为(B,T)里面的元素为音素的pitch，但在推理阶段是None\n
        输出为：prediction, embedding，其中prediction的形状为(Batchsize,T)，其中元素为log(预测的音素的帧的时长+1)，embedding的形状为(B,T,256) ,表示pitch的嵌入"""

        #mask就是用来屏蔽填充的
        prediction = self.pitch_predictor(x, mask)
        """prediction的形状为(Batchsize,T)，其中元素为log(预测的音素的帧的时长+1)"""
        if target is not None:
            embedding = self.pitch_embedding(target.unsqueeze(-1)) if self.use_conv_embedding \
                else self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
            """(B,T,256) ,表示pitch的嵌入"""
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(prediction.unsqueeze(-1)) if self.use_conv_embedding \
                else self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    #跟上面的get_pitch_embedding方法一摸一样
    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding =  self.energy_embedding(target.unsqueeze(-1)) if self.use_conv_embedding \
                else self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(prediction.unsqueeze(-1)) if self.use_conv_embedding \
                else self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        speaker_embedding, #(B,256)
        emotion_embedding,  #(B,256)
        context_encoding,  #(B,256)   #是最近的特征融合的结果
        text, #(B,T,256)
        text_embedding, #(B,T,256)
        src_len,   #(B,) 存放音频的实际音素序列长度
        src_mask, #(B,T)
        mel,  #(B,D,256)
        mel_len,  #(B,) 存放音频的实际帧长
        mel_mask=None,
        max_len=None,
        pitch_target=None, #(B, D)， 每帧的pitch 数值。D为batch中最大帧数
        energy_target=None,  #(B, D)， 每帧的energy 数值。D为batch中最大帧数
        duration_target=None,
        attn_prior=None,  #(B,D,T)
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None,
    ):
        x = text
        #三个if用来融合说话人，情感和上下文到文本音素embedding上
        if speaker_embedding is not None:
            x = x + speaker_embedding.unsqueeze(1).expand(
                -1, text.shape[1], -1
            )
        if emotion_embedding is not None:
            x = x + emotion_embedding.unsqueeze(1).expand(
                -1, text.shape[1], -1
            )
        # x_dur = x.clone()
        if context_encoding is not None:
            x = x + context_encoding.unsqueeze(1).expand(
                -1, text.shape[1], -1
            )

        log_duration_prediction = self.duration_predictor(x, src_mask)
        """log_duration_prediction形状为(B,T),存放预测的音素的（帧数+1）的log值"""
        # log_duration_prediction = self.duration_predictor(x_dur, src_len, context_encoding, src_mask)
        duration_rounded = torch.clamp(
            (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
            min=0,
        )
        """duration_rounded形状为(B,T),存放预测的音素的帧数"""

        # Trainig of unsupervised duration modeling
        attn_soft, attn_hard, attn_hard_dur, attn_logprob = None, None, None, None
        if attn_prior is not None:
            assert self.learn_alignment and duration_target is None and mel is not None
            attn_soft, attn_logprob = self.aligner(
                mel.transpose(1, 2),
                text_embedding.transpose(1, 2),
                src_mask.unsqueeze(-1),
                attn_prior.transpose(1, 2),
                speaker_embedding,
                emotion_embedding,
            )
            """attn_soft是一个B x 1 x D x T 的 attention 矩阵， 这个注意力矩阵是个概率，在(B,1,D,T)上对最后一维用softmax，(对于确定的b和d)attn[b,0,d,i] 就可以理解为“第 d 帧对应（或“对齐”到）第 i 个音素的概率
             attn_logprob是其取对数"""
            attn_hard = self.binarize_attention_parallel(attn_soft, src_len, mel_len)  #返回(B,1,D,T)，在GPU上，其中存储了batch中每个音频的帧音素的对齐路径信息(用0，1来表示) 并且包含padding的全0部分
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]
            """attn_hard_dur的形状 (B, T)，表示每个音素被模型预测的帧数总和(这里是模型经过先验注意力得到的时长的伪标签)。"""
        attn_out = (attn_soft, attn_hard, attn_hard_dur, attn_logprob)  #含有四个元素的元组

        # Note that there is no pre-extracted phoneme-level variance features in unsupervised duration modeling.
        # Alternatively, we can use attn_hard_dur instead of duration_target for computing phoneme-level variances.
        output_1 = x.clone()  #此时的output1，是音素嵌入序列融合了说话人，情感和上下文
        if self.pitch_feature_level == "phoneme_level":
            if attn_prior is not None:
                pitch_target = self.get_phoneme_level_pitch(attn_hard_dur, src_len, pitch_target) #pitch_target的shape为(B,T) T为batch中音频的最大音素序列长度，每个元素为对应音素的音素级pitch就是(音素对应的) 插值过的帧级pitch取平均
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, src_mask, p_control)
            output_1 = output_1 + pitch_embedding   #添加pitch信息( b,t,256)
        if self.energy_feature_level == "phoneme_level":
            if attn_prior is not None:
                energy_target = self.get_phoneme_level_energy(attn_hard_dur, src_len, energy_target)
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, src_mask, e_control)
            output_1 = output_1 + energy_embedding   #添加energy信息( b,t,256)
        x = output_1.clone()

        # Upsampling from src length to mel length
        #无监督对齐训练阶段（用 Attention+MAS 造伪标签）
        if attn_prior is not None: # Trainig of unsupervised duration modeling
            if step < self.binarization_start_steps:
                # 把“软对齐”的注意力分布 A_soft直接用来把音素级特征x“映射”到每一帧上(变成帧级256维特征)：
                A_soft = attn_soft.squeeze(1)
                x = torch.bmm(A_soft,x)  #这个是矩阵乘法，此时x=(B,D,256)
            else:
                # 这里的x是使用长度调节器之后的，由(B,T,256)->(B,D,256) D为最大帧数
                #mel_len为一个long类型的张量，张量形状为(batchsize,) 是一个一维张量，存放的是一个batch中的音频帧长
                x, mel_len = self.length_regulator(x, attn_hard_dur, max_len)
            duration_rounded = attn_hard_dur
        elif duration_target is not None: # Trainig of supervised duration modeling
            assert not self.learn_alignment and attn_prior is None
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else: # Inference
            #在推理阶段做的，用的是duration_predictor类预测的音素帧数
            assert attn_prior is None and duration_target is None
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        output_2 = x.clone()  #这里的x是经过了长度调节器扩展之后的，将音素按帧数扩展
        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, mel_mask, p_control)
            output_2 = output_2 + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, mel_mask, e_control)
            output_2 = output_2 + energy_embedding
        x = output_2.clone()

        return (
            x,  #(B,D,256)
            pitch_target, #pitch_target的shape为(B,T) T为batch中音频的最大音素序列长度，每个元素为对应音素的音素级pitch就是(音素对应的) 插值过的帧级pitch取平均
            pitch_prediction,  #prediction的形状为(Batchsize,T)，其中元素为模型预测的音素级的pitch
            energy_target,  #同理跟pitch_target一样，只不过是energy
            energy_prediction,  #同理跟pitch_prediction一样，只不过是energy
            log_duration_prediction, #log_duration_prediction形状为(B,T),存放音素的（帧数+1）的log值
            duration_rounded,  #duration_rounded形状为(B,T),存放音素的帧数
            mel_len,  #mel_len为一个long类型的张量，张量形状为(batchsize,) 是一个一维张量，存放的是一个batch中的音频帧长
            mel_mask,#一个形状为 (B, D) 的布尔掩码张量 mask，D为一个batch中的最大帧长，用来屏蔽掉那些“填充”出来的无效位置
            attn_out,  #attn_out = (attn_soft, attn_hard, attn_hard_dur, attn_logprob) ，含有四个元素的元组
        )


class AlignmentEncoder(torch.nn.Module):
    """对齐编码器，用于无监督时长建模，通过计算音素与帧之间的对齐信息（注意力），得到注意力矩阵(B,1,D,T)。\n
    这个注意力矩阵是个概率，在(B,1,D,T)上对最后一维用softmax，(对于确定的b和d)attn[b,0,d,i] 就可以理解为“第 d 帧对应（或“对齐”到）第 i 个音素的概率”。"""

    def __init__(self,
                n_mel_channels,   # mel 频谱的通道数     80
                n_att_channels,   # 对齐空间的通道数（投影后的维度）  80
                n_text_channels,   # 文本编码的通道数   256
                temperature,     # 温度系数，用于控制 attention 的平滑度  0.0005
                multi_speaker,   # 是否为多说话人场景   true
                multi_emotion):  # 是否为多情感场景     true
        super().__init__()
        self.temperature = temperature
        # Softmax 用于计算最终 attention 概率分布，作用在最后一个维度（T2，即文本长度）上,dim从左到右为0，1，2，3....，
        self.softmax = torch.nn.Softmax(dim=3)  #结果就是固定前三个维度，最后一个维度做softmax运算从最小到最大和为1
        # LogSoftmax 用于计算对数概率，后续与先验相加
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        # 文本特征投影层（Key Projection），对文本编码 keys 进行投影，将维度从(Batchsize,256,T) 投影到 (Batchsize,80,T) ，其中T代表当前batch的最大音素序列长度
        self.key_proj = nn.Sequential(
            ConvNorm(
                n_text_channels,
                n_text_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain='relu'
            ),
            torch.nn.ReLU(),
            ConvNorm(
                n_text_channels * 2,
                n_att_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        # Mel特征投影层（Query Projection）对 mel 编码 queries 进行投影，将维度从 n_mel_channels 80 投影到 n_att_channels 80
        self.query_proj = nn.Sequential(
            ConvNorm(
                n_mel_channels,
                n_mel_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain='relu',
            ),
            torch.nn.ReLU(),
            ConvNorm(
                n_mel_channels * 2,
                n_mel_channels,
                kernel_size=1,
                bias=True,
            ),
            torch.nn.ReLU(),
            ConvNorm(
                n_mel_channels,
                n_att_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        # 多说话人投影层（可选），如果是多说话人场景，则需要额外的投影，将说话人嵌入加到文本和 mel 上，说话人嵌入是Batch*256维的好像
        if multi_speaker:
            self.key_spk_proj = LinearNorm(n_text_channels, n_text_channels)
            self.query_spk_proj = LinearNorm(n_text_channels, n_mel_channels)
        # 如果是多情感场景，同理，对情感嵌入进行投影
        if multi_emotion:
            self.key_emo_proj = LinearNorm(n_text_channels, n_text_channels)
            self.query_emo_proj = LinearNorm(n_text_channels, n_mel_channels)

    def forward(self, queries, keys, mask=None, attn_prior=None, speaker_embed=None, emotion_embed=None):
        """
               前向传播过程：
               Args:
                   queries (torch.tensor): B x 256 x duration 的张量，通常是 mel 频谱，C为通道数，T1为时间步数(帧长)
                   keys (torch.tensor): B x 256 x T 的张量，通常是文本编码数据，T为一个batch的最大音素序列长度
                   mask (torch.tensor): B x T x 1
                   attn_prior (torch.tensor): attention 先验，用于无监督对齐时辅助计算，注意力先验矩阵 (B,音素个数，帧数)
                   speaker_embed (torch.tensor): 多说话人时的说话人嵌入，尺寸 B x 256？
                   emotion_embed (torch.tensor): 多情感时的情感嵌入，尺寸 B x 256？
               Output:
                   attn (torch.tensor): B x 1 x D x T 的 attention 矩阵， 这个注意力矩阵是个概率，在(B,1,D,T)上对最后一维用softmax，(对于确定的b和d)attn[b,0,d,i] 就可以理解为“第 d 帧对应（或“对齐”到）第 i 个音素的概率
                   attn_logprob (torch.tensor): 上面attn对应的对数概率注意力矩阵
        """
        # 多说话人特征融合
        if speaker_embed is not None:
            """
            speaker_embed 经过 unsqueeze(1) 和 expand 后，形状为 (B, T2, C)。
            """

            # 将说话人嵌入投影到文本特征空间并加到keys
            keys = keys + self.key_spk_proj(speaker_embed.unsqueeze(1).expand(
                -1, keys.shape[-1], -1
            )).transpose(1, 2)   #expand 方法用于将张量的某个或某些维度进行扩展，扩展的方式是通过复制原有元素来实现的。expand 方法接受一个元组作为参数，元组中的每个元素对应张量的一个维度，指定该维度要扩展到的大小。-1 表示该维度保持原有的大小不变。

            # 将说话人嵌入投影到Mel特征空间并加到queries
            queries = queries + self.query_spk_proj(speaker_embed.unsqueeze(1).expand(
                -1, queries.shape[-1], -1
            )).transpose(1, 2)
        """逻辑同上"""
        if emotion_embed is not None:
            keys = keys + self.key_emo_proj(emotion_embed.unsqueeze(1).expand(
                -1, keys.shape[-1], -1
            )).transpose(1, 2)
            queries = queries + self.query_emo_proj(emotion_embed.unsqueeze(1).expand(
                -1, queries.shape[-1], -1
            )).transpose(1, 2)
        """特征投影"""
        keys_enc = self.key_proj(keys)  # B x 80 x T
        queries_enc = self.query_proj(queries)      #B x 80 x duration

        # Simplistic Gaussian Isotopic Attention
        # 高斯同构注意力计算
        # 计算欧氏距离平方：(queries_enc - keys_enc)^2
        #None就是 np.newaxis 的别名，用来在那个位置“插入”一个长度为 1 的新维度
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2  # 维度为B x 80 x duration x T
        """attn是经过融合了网络预测和先验信息后，并且经过掩码之后的注意力矩阵"""
        #这样目前的attn[b, c, d, t] 存的是第 c 通道在第 d 帧与第 t 时刻的平方差

        attn = -self.temperature * attn.sum(1, keepdim=True)      # 对特征维度求和并缩放，经过sum之后，变成了各通道平方差(音素序列与帧的平方差)的累加 ,形状为(B,1,D,T)

        # 融合注意力先验（监督对齐时使用）
        if attn_prior is not None:
            #print(f"AlignmentEncoder \t| mel: {queries.shape} phone: {keys.shape} mask: {mask.shape} attn: {attn.shape} attn_prior: {attn_prior.shape}")
            # 将先验概率转换为对数空间并相加
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-8)
            #print(f"AlignmentEncoder \t| After prior sum attn: {attn.shape}")

        attn_logprob = attn.clone()
        """attn_logprob是融合了网络预测和先验信息后的注意力矩阵，形状为(B,1,D,T)"""

        # 应用掩码（忽略填充部分）
        if mask is not None:
            attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2), -float("inf"))  #-float("inf")为负无穷

        # 计算注意力矩阵（softmax）
        attn = self.softmax(attn)  # softmax along T  #沿T维度归一化
        """attn是经过融合了网络预测和先验信息后，并且经过掩码之后的注意力矩阵"""
        return attn, attn_logprob


class LengthRegulator(nn.Module):
    """forward函数为：输入x为（B，T，256），输入duration为(B,T),max_len，\n
                输出为out，将音素文本嵌入按照持续时间扩展，最后cat成一个大张量，shape为(B,最大帧长，256)，其中最大帧长为一个batch中音频的最大帧长\n
                第二个输出为一个long类型的张量，张量形状为(batchsize,)是一个一维张量，存放的是一个batch中的音频帧长"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        """输入x为（B，T，256），输入duration为(B,T),max_len，\n
        输出为out，将音素文本嵌入按照持续时间扩展，最后cat成一个大张量，shape为(B,最大帧长，256)，其中最大帧长为一个batch中音频的最大帧长\n
        第二个输出为一个long类型的张量，张量形状为(batchsize,)是一个一维张量，存放的是一个batch中的音频帧长"""
        output = list()
        mel_len = list()  #存放的是一个batch中的音频帧长，list长度为batchsize
        for batch, expand_target in zip(x, duration):
            #这个for循环相当于对一个batch里的数据迭代，对batch中的每个音频处理
            #batch: Tensor, shape=(T, 256)  ，expand_target: Tensor, shape=(T,)
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)  # 会把列表里的每个整数依次拷贝到一个新的张量里，默认 dtype 是 torch.int64

    def expand(self, batch, predicted):
        """将某个音频的音素文本嵌入按照持续时间扩展，shape为(实际帧长，256)"""
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()   #.item() 是一个很常见的方法，用来把只含单个元素的 0 维张量（tensor([])）里的值“取出来”变成一个纯 Python 标量（int 或 float）
            out.append(vec.expand(max(int(expand_size), 0), -1))    # vec.expand(n, -1) 会把 vec 从 (256,) 复制成 (n, 256)  ,如果n为0，相当于添加了一个空张量(（0，256）但是在后面的torch.cat会直接丢弃)
        out = torch.cat(out, 0)   #torch.cat(out, 0) 的作用，就是把 Python 列表 out 里所有的张量，沿着第 0 个维度（也就是“行”方向，第0维数字变化）拼接成一个大张量。

        return out

    def forward(self, x, duration, max_len):
        """输入x为（B，T，256），输入duration为(B,T),max_len，\n
                输出为out，将音素文本嵌入按照持续时间扩展，最后cat成一个大张量，shape为(B,最大帧长，256)，其中最大帧长为一个batch中音频的最大帧长\n
                第二个输出为一个long类型的张量，张量形状为(batchsize,)是一个一维张量，存放的是一个batch中的音频帧长"""
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len

#没用到（DurationPredictor会使用LayerCondFFTBlock，而LayerCondFFTBlock会使用StyleAdaptiveLayerNorm）
class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config, positive_out=True):
        super(DurationPredictor, self).__init__()

        self.d_model = model_config["transformer"]["encoder_hidden"]
        self.d_hidden = model_config["variance_predictor"]["cond_dur_hidden"]

        self.max_seq_len = model_config["max_seq_len"]
        n_position = self.max_seq_len + 1
        n_head = model_config["variance_predictor"]["cond_dur_head"]
        d_w = self.d_hidden
        d_k = d_v = d_w // n_head
        d_inner = model_config["variance_predictor"]["conv_filter_size"]
        kernel_size = model_config["variance_predictor"]["conv_kernel_size"]
        dropout = model_config["variance_predictor"]["cond_dur_dropout"]

        self.cond_prj = LinearNorm(self.d_model, self.d_hidden)
        self.input_prj = nn.Sequential(
            ConvNorm(self.d_model, self.d_hidden, transpose=True),
            Swish(),
            LinearNorm(self.d_hidden, self.d_hidden),
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.d_hidden).unsqueeze(0),
            requires_grad=False,
        )
        self.layer_stack = nn.ModuleList(
            [
                LayerCondFFTBlock(
                    self.d_hidden, d_w, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(model_config["variance_predictor"]["cond_dur_layer"])
            ]
        )
        self.out = nn.Sequential(
            ConvNorm(self.d_hidden, 1, transpose=True),
            nn.ReLU() if positive_out else Swish(),
        )

    def forward(self, h_text, seq_len, h_context, mask):
        batch_size, max_len = h_text.shape[0], h_text.shape[1]

        # Input
        cond_g = self.cond_prj(h_context.unsqueeze(1)) # [B, 1, H]
        h_text = self.input_prj(h_text) # [B, seq_len, H]

        # Mask
        h_text = h_text.masked_fill(mask.unsqueeze(-1), 0)
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # Positional Encoding
        if not self.training and h_text.shape[1] > self.max_seq_len:
            output = h_text + get_sinusoid_encoding_table(
                h_text.shape[1], self.d_hidden
            )[: h_text.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                h_text.device
            )
        else:
            output = h_text + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        # Conditioned Duration Prediction
        for layer in self.layer_stack:
            output, _ = layer(
                output, cond_g, mask=mask, slf_attn_mask=slf_attn_mask
            )
        output = self.out(output).squeeze(-1)

        return output

#没用到
class LayerCondFFTBlock(nn.Module):
    """ Layer Conditioning FFTBlock """

    def __init__(self, d_model, d_w, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(LayerCondFFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, layer_norm=False)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout, layer_norm=False
        )
        self.layer_norm_1 = StyleAdaptiveLayerNorm(d_w, d_model)
        self.layer_norm_2 = StyleAdaptiveLayerNorm(d_w, d_model)

    def forward(self, enc_input, cond_g, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.layer_norm_1(enc_output, cond_g)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = self.layer_norm_2(enc_output, cond_g)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn

#没用到
class StyleAdaptiveLayerNorm(nn.Module):
    """ Style-Adaptive Layer Norm (SALN) """

    def __init__(self, w_size, hidden_size, bias=False):
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.affine_layer = LinearNorm(
            w_size,
            2 * hidden_size, # For both b (bias) g (gain)
            bias,
        )

    def forward(self, h, cond_g):
        """
        h --- [B, T, H_m]
        cond_g --- [B, 1, H_w]
        o --- [B, T, H_m]
        """

        # Normalize Input Features
        mu, sigma = torch.mean(h, dim=-1, keepdim=True), torch.std(h, dim=-1, keepdim=True)
        y = (h - mu) / sigma # [B, T, H_m]

        # Get Bias and Gain
        b, g = torch.split(self.affine_layer(cond_g), self.hidden_size, dim=-1)  # [B, 1, 2 * H_m] --> 2 * [B, 1, H_m]

        # Perform Scailing and Shifting
        o = g * y + b # [B, T, H_m]

        return o

#真实的持续时间标签（即每个音素的实际时长）通过使用音素对齐工具（如蒙特利尔强制对齐器 Montreal Forced Aligner，或基于 HMM/DNN 的对齐模型）将文本与音频波形对齐。得到每个音素在音频中的起始时间和结束时间，从而计算出持续时间：
class VariancePredictor(nn.Module):
    """ forward函数传入参数为encoder_output, mask,其中mask的形状为(Batchsize,T)，encoder_output参数来自于transformers.py中的TextEncoder的输出结果，形状为(Batchsize,T，256)，T为当前batch中音素序列的最大长度，(因为不满足的在dataset中会被扩充) \n
        输出结果的形状为(Batchsize,T)，是对encoder_output进行处理(生成帧长)， T为当前batch中音素序列的最大长度（max_len），其中元素为log(预测的音素的帧的时长+1)\n
        """
    #这些卷积操作就是为了拟合计算出log(预测的音素的帧的时长+1)
    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]  #256
        self.filter_size = model_config["variance_predictor"]["filter_size"]  #256
        self.kernel = model_config["variance_predictor"]["kernel_size"]   #3
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]  #256
        self.dropout = model_config["variance_predictor"]["dropout"] # 0.5

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        ConvNorm(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            stride=1,
                            padding=(self.kernel - 1) // 2,
                            dilation=1,
                            transpose=True,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),  #对于输入张量的每个元素，有0.5的概率为0，为了保证期望不变，对剩下的元素做放大处理
                    (
                        "conv1d_2",
                        ConvNorm(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            stride=1,
                            padding=1,
                            dilation=1,
                            transpose=True,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            #mask(布尔/二值张量)，形状都为(Batchsize,T),T为当前batch中音素序列的最大长度，
            out = out.masked_fill(mask, 0.0)  #它会返回一个新的张量，和 out 形状完全一样，如果 mask[i, j] == True，就把 out[i, j] 的值改成 0.0；

        return out

#原本的dailytalk的上下文编码器其实就是复现concss中的Textual Context Module中的Utterance‑Level Module（Coarse‑grained，粗粒度编码器）
class ConversationalContextEncoder(nn.Module):
    """输入：text_emb(B,512); speaker(B,);history_text_emb(B,10,512)?; history_speaker(B,10):历史句子的说话人 ID 列表;history_lens(B,):实际提供了多少历史句;\n
       输出：(B,256)对话上下文嵌入"""

    def __init__(self, preprocess_config, model_config):
        super(ConversationalContextEncoder, self).__init__()
        d_model = model_config["transformer"]["encoder_hidden"]  #256
        d_cont_enc = model_config["history_encoder"]["context_hidden"]  #128
        num_layers = model_config["history_encoder"]["context_layer"]  #2
        dropout = model_config["history_encoder"]["context_dropout"]  #0.2
        self.text_emb_size = model_config["history_encoder"]["text_emb_size"]  #512
        self.max_history_len = model_config["history_encoder"]["max_history_len"]  #10

        self.text_emb_linear = nn.Linear(self.text_emb_size, d_cont_enc)  #线性层从512到128
        self.speaker_linear = nn.Linear(d_model, d_cont_enc)  #从256到128
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            ),
            "r",
        ) as f:
            n_speaker = len(json.load(f))
            """n_speaker=2"""
        self.speaker_embedding = nn.Embedding(
            n_speaker,
            model_config["transformer"]["encoder_hidden"], #256
        )

        #256->128
        self.enc_linear = nn.Sequential(
            nn.Linear(2*d_cont_enc, d_cont_enc),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            input_size=d_cont_enc,  #128
            hidden_size=d_cont_enc,  #128
            num_layers=num_layers,  #2
            batch_first=True,
            dropout=dropout,  #0.2
            bidirectional=True
        )
        """初始化一个多层、双向的 GRU，用来对序列做上下文建模,输入 x：形状 (B, T, 128),输出 out：形状 (B, T, 256),因为 bidirectional=True，正向和反向各输出 128 维，再拼在一起，成了 2×128=256 维。"""
        self.gru_linear = nn.Sequential(
            nn.Linear(2*d_cont_enc, d_cont_enc),
            nn.ReLU()
        )  #256->128

        #128->256
        self.context_linear = nn.Linear(d_cont_enc, d_model)
        self.context_attention = SLA(d_model)  #SLA: 一层 Self‐Attention，用来把序列聚合成 [B,256] 向量


    def forward(self, text_emb, speaker, history_text_emb, history_speaker, history_lens):
        """输入：1.text_emb(B,512)?; 2.speaker(B,);3.history_text_emb(B,10,512)?; 4.history_speaker(B,10):历史句子的说话人 ID 列表;5.history_lens(B,):实际提供了多少历史句;\n
               输出：(B,256)对话上下文嵌入"""
        history_masks = get_mask_from_lengths(history_lens, self.max_history_len) #history_masks为(B,10)

        # Embedding
        history_text_emb = torch.cat([history_text_emb, text_emb.unsqueeze(1)], dim=1) #把当前句append 到历史尾部，形成长为 10+1 的序列(B,11,512)
        history_text_emb = self.text_emb_linear(history_text_emb) #(B,11,512)变为(B,11,128)
        history_speaker = torch.cat([history_speaker, speaker.unsqueeze(1)], dim=1)  #把当前句说话人id append 到历史尾部，形成长为 10+1 的序列(B,11)
        history_speaker = self.speaker_linear(self.speaker_embedding(history_speaker)) #说话人ID → (B, 11,  256) → 线性变换 → (B, 11, 128)

        #文本 & 说话人 两路特征拼接
        history_enc = torch.cat([history_text_emb, history_speaker], dim=-1)  # history_enc经过拼接变为(B,11,256)
        history_enc = self.enc_linear(history_enc)  #(B,11,256)->(B,11,128)

        # Split, enc_current为(B, 10, 128)，而enc_past为(B,1,128)
        enc_current, enc_past = torch.split(history_enc, self.max_history_len, dim=1) #就是把张量 x 沿着第1维切成若干块，每块长度为max_history_len=10，最后一块如果长度不够就只包含剩下的元素。

        # GRU
        enc_current = self.gru_linear(self.gru(enc_current)[0])  #self.gru(enc_current)[0],返回(B,10,256),调用gru返回的是二元组（output, h_n），线性变换又变成了(B,10,128)
        enc_current = enc_current.masked_fill(history_masks.unsqueeze(-1), 0) #根据mask填充0

        # Encoding
        context_enc = torch.cat([enc_current, enc_past], dim=1) #再次拼接当前句变成(B, 11, 128)
        #linear操作先从(B, 11, 128)->(B, 11, 256)
        context_enc = self.context_attention(self.context_linear(context_enc)) # [B, 256]

        return context_enc


class SLA(nn.Module):
    """SLA: 一层 Self‐Attention，用来把序列聚合成 [B,256] 向量"""
    #d_enc=256
    def __init__(self, d_enc):
        super(SLA, self).__init__()
        self.linear = nn.Linear(d_enc, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoding, mask=None):

        attn = self.linear(encoding)  #(b,11,256)->(b,11,1)变成得分注意力
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1), -np.inf)
            aux_mask = (attn == -np.inf).all(self.softmax.dim).unsqueeze(self.softmax.dim)
            attn = attn.masked_fill(aux_mask, 0) # Remove all -inf along softmax.dim
        #(b,11,1)首先softmax，然后再转置
        score = self.softmax(attn).transpose(-2, -1) # [B, 1, 11]
        fused_rep = torch.matmul(score, encoding).squeeze(1) # [B, 256]

        return fused_rep




#Phoneme‑Level Textual Context Module (TPM)  细粒度文本上下文编码器
class TextPhonemeLevelModule(nn.Module):
    """
    Token‑level (phoneme) context encoder using BERT + cross-attention.

    功能:
      1. 对历史最多 max_history 条 utterance 的 phoneme ID 序列进行 BERT 编码
      2. 融合每条历史 utterance 对应的说话人 embedding
      3. 位置编码、卷积融合后将所有历史 token 拼接
      4. 用 cross-attention 将 tts_query 与历史 token 对齐

    参数:
      bert_model_name (str): 预训练 BERT 名称
      d_model (int):         BERT 输出与内部维度
      n_head (int):          多头注意力头数
      conv_kernel (int):     1D 卷积核大小
      max_history (int):     最大历史 utterance 数
    """
    def __init__(self,
                 bert_model_name: str = "xxx",
                 d_model: int = 256,
                 n_head: int = 4,
                 conv_kernel: int = 3,
                 max_history: int = 10):
        super().__init__()
        self.max_history = max_history

        # 1) BERT 编码
        self.bert = DistilBertModel.from_pretrained(bert_model_name)   #.to(device)

        # 冻结所有参数
        for p in self.bert.parameters():
            p.requires_grad = False

        # 2) 历史说话人投影: 输入 (B, H, d_model) → 输出相同维度
        self.spk_proj = nn.Linear(d_model, d_model)
        # 3) 将BERT输出从768维映射到d_model维度
        self.h_proj = nn.Linear(768, d_model)
        # 4) 1D 卷积，用于融合局部 token 上下文
        self.conv = ConvNorm(d_model, d_model,
                             kernel_size=conv_kernel,
                             stride=1,
                             padding=(conv_kernel-1)//2)
        # 5) Cross-Attention
        self.cross_attn = MultiHeadAttention(n_head, d_model,
                                             d_k=d_model//n_head,
                                             d_v=d_model//n_head)
        # 6) 前馈网络
        self.ffn = PositionwiseFeedForward(d_model,
                                           d_model*4,
                                           kernel_size=[conv_kernel, conv_kernel])

    def forward(self,
                history_inputs: torch.Tensor,
                history_masks: torch.Tensor,
                tts_query: torch.Tensor,
                history_speaker_emb: torch.Tensor,
                max_hist_tokens: int,
                id=1,) -> torch.Tensor:
        """
        输入:
        H=10
        - history_inputs:    (B, H, T)   phoneme ID
        - history_masks:     (B, H, T)   phoneme mask，true代表真实，false代表padding
        - tts_query:         (B, L, d_model)
        - history_speaker_emb: (B, H, d_model)  每条历史 utterance 的说话人向量
        - max_hist_tokens (int): 历史 token 最大长度 T
        返回:
        - out:               (B, L, d_model)
        """
        # --- 强制把 numpy 转成 torch.Tensor 并放到同一个 device 上 ---
        if not isinstance(history_inputs, torch.Tensor):
            history_inputs = torch.from_numpy(history_inputs).long().to(tts_query.device)
        if not isinstance(history_masks, torch.Tensor):
            history_masks = torch.from_numpy(history_masks).float().to(tts_query.device)

        B, H, T = history_inputs.shape   #H=10

        # —— 1) flatten utterance: (B*H, T)
        flat_inputs = history_inputs.reshape(B*H, T)
        flat_masks = history_masks.reshape(B*H, T)


        try:
            save_dir = os.path.join("xxx", str(id))
            save_path = os.path.join(save_dir, "bert_features.pt")
            h=torch.load(save_path)

        except FileNotFoundError:
            bert_out = self.bert(input_ids=flat_inputs,
                                 attention_mask=flat_masks)
            h = bert_out.last_hidden_state  # (B*H, T, 768)
            # 把 BERT 可能在 padding 处生产的 NaN 全关掉
            h = torch.nan_to_num(h, nan=0.0)

        # 投影到 d_model 维度
        h = self.h_proj(h)  # (B*H, T, d_model)

        # —— 3) 融合历史说话人 embedding
        spk_flat = history_speaker_emb.reshape(B*H, -1)
        spk_proj = self.spk_proj(spk_flat).unsqueeze(1)  # (B*H, 1, d_model)
        # add speaker proj to each token
        x = h + spk_proj

        # —— 4) 添加位置编码
        pos_emb = get_sinusoid_encoding_table(max_hist_tokens, h.size(-1)).to(h.device)
        pos = pos_emb[:T].unsqueeze(0).expand(B*H, -1, -1)
        x = x + pos

        #置padding部分为0
        x = x * flat_masks.unsqueeze(-1).type_as(x)  # (B*H, T, d_model)
        # —— 5) Conv 局部融合 → (B*H, T, d_model)
        x = self.conv(x.transpose(1,2)).transpose(1,2)

        # —— 6) 重塑并拼接所有历史 tokens → (B, H*T, d_model)
        x = x.reshape(B, H*T, -1)

        # —— 7) Cross-Attention 对齐 tts_query 与历史 tokens
        full_mask_bool = flat_masks.reshape(B, H * T)  # 将 (B*H, T) 映射至 (B, H*T)
        attn_mask = (~full_mask_bool).unsqueeze(1).to(torch.bool)   # 扩展为 (B, 1, H*T)
        attn_mask=self.fix_attn_mask(attn_mask,Lq=tts_query.size(1))
        attn_out, _ = self.cross_attn(tts_query, x, x,mask=attn_mask)
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)

        # —— 8) FFN + 残差 → out
        out = self.ffn(attn_out)

        #【优化】强制清洗Turn = 0的样本
        #    如果一个样本完全没有历史 (full_mask_float全是0)，
        #    虽然 fix_attn_mask 保证了不崩，但会算出 Bias 噪音。
        #    这里强制把结果乘 0，保证无历史=无影响。
        valid_rows = (full_mask_bool.sum(dim=-1) > 0).float()  # (B,)
        out = out * valid_rows.view(B, 1, 1).type_as(out)
        return out

    def fix_attn_mask(self,attn_mask: torch.Tensor, Lq: int) -> torch.Tensor:
        """
        修复 attention mask，确保每个 query 至少能 attend 到一个 key。

        参数:
        - attn_mask: BoolTensor, 形状 (B, 1, S)，True 表示被屏蔽（padding）
        - Lq: int, query 序列长度

        返回:
        - new_mask: BoolTensor, 形状 (B, 1, S)，修正后的 mask
        """
        B, _, S = attn_mask.shape
        # 1) squeeze -> (B, S)
        base_mask = attn_mask.squeeze(1)  # (B, S)

        # 2) expand -> (B, Lq, S)
        mask_expanded = base_mask.unsqueeze(1).expand(-1, Lq, -1).clone()  # (B, Lq, S)

        # 3) 找出哪些行全被屏蔽了
        all_masked = (mask_expanded.sum(dim=-1) == S)  # (B, Lq)；True 表示这一行全 True

        if all_masked.any():
            # 拿到所有 (i,j) 坐标
            idx = all_masked.nonzero(as_tuple=False)  # Tensor of shape (K, 2): [[i0, j0], [i1, j1], ...]
            # 把这些行的 key=0 置为 False（表示“不屏蔽”）
            mask_expanded[idx[:, 0], idx[:, 1], 0] = False

        # 4) 收回 (B,1,S)
        new_mask = mask_expanded[:, :1, :].to(attn_mask.dtype)
        return new_mask


# ====================================================
# 2. 音频粗粒度模块 AcousticCoarseModule
#     T = 每条 utterance 的采样长度（已 pad/truncate）,根据采样频率得到的音频总共采样的个数
#    supports batch_history_wavs: (B, H, T)
# ====================================================
class AcousticCoarseModule(nn.Module):
    """
    粗粒度声学上下文编码器 (Utterance‑Level Acoustic Context Encoder)

    功能:
      1. 对历史最多 max_history 条音频 (每条长度 T) 提取全局向量
      2. 使用 Wav2Vec2 从每条历史 utterance 中抽取帧级特征并做 mean‑pooling
      3. 用 GRU 聚合这 max_history 条历史 utterance 的时序上下文
      4. 基于聚合后的 GRU 输出做注意力加权，得到固定维度的上下文表示

    参数:
      wav2vec_model_name (str): HuggingFace Wav2Vec2 预训练模型名称
      audio_dim          (int): Wav2Vec2 输出特征维度 (d_model)
      gru_hidden         (int): GRU 隐藏层维度
      out_dim            (int): 最终上下文向量维度 (本例设 256)
      max_history        (int): 最大历史 utterance 条数 (默认 10)

    输入:
    T = 每条 utterance 的采样长度（已 pad/truncate）,根据采样频率得到的音频总共采样的个数
      batch_history_wavs  (Tensor[B, H, T]): pad 后的历史音频波形，H≤max_history
      batch_wav_masks     (Tensor[B, H, T]): pad mask (1=真实帧, 0=pad)

    输出:
      Tensor[B, out_dim]: 每个样本的 utterance‑level 上下文向量
    """
    def __init__(self,
                 wav2vec_model_name: str   = "xxx/wav2vec2-base-100h",
                 audio_dim:          int   = 768,
                 gru_hidden:         int   = 128,
                 out_dim:            int   = 256,
                 max_history:        int   = 10):
        super().__init__()
        self.max_history = max_history

        # 1) Wav2Vec2 用于提帧级特征
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            wav2vec_model_name
        )   #.to(device)

        #冻结所有参数
        for p in self.wav2vec.parameters():
            p.requires_grad = False

        self.d_model = audio_dim   #768

        # 2) GRU 聚合多条历史 utterance
        self.gru     = nn.GRU(audio_dim, gru_hidden, batch_first=True)
        # 3) 注意力与线性映射
        self.attn    = nn.Linear(gru_hidden, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.linear  = nn.Linear(gru_hidden, out_dim)

    def forward(self,
                batch_history_wavs: torch.Tensor,
                batch_wav_masks:   torch.Tensor,
                id=1,) -> torch.Tensor:
        """
        T = 每条 utterance 的采样长度（已 pad/truncate）,根据采样频率得到的音频总共采样的个数
        batch_history_wavs: (B, H, T)
        batch_wav_masks:   (B, H, T)
        returns H_coarse:  (B, out_dim)
        """
        device = next(self.parameters()).device
        # --- 强制把 numpy 转成 torch.Tensor 并放到同一个 device 上 ---
        if not isinstance(batch_history_wavs, torch.Tensor):
            batch_history_wavs = torch.from_numpy(batch_history_wavs).float().to(device)
        if not isinstance(batch_wav_masks, torch.Tensor):
            batch_wav_masks = torch.from_numpy(batch_wav_masks).float().to(device)
        B, H, T = batch_history_wavs.shape

        # —— 1) Flatten history utterances → (B*H, T)
        flat_wavs = batch_history_wavs.reshape(B * H, T)
        flat_mask = batch_wav_masks.reshape(B * H, T)

        try:
            save_dir = os.path.join("xxx", str(id))
            save_path = os.path.join(save_dir, "wav2vec_features.pt")
            feats=torch.load(save_path)
            # padding部分置0
            token_mask = flat_mask.unsqueeze(1).float()  # (B*H, 1, T)
        except FileNotFoundError:
            with torch.no_grad():
                feats = self.wav2vec(flat_wavs, attention_mask=flat_mask)[0]
            # 把 Wav2Vec2 可能在 padding 处生产的 NaN 全关掉
            feats = torch.nan_to_num(feats, nan=0.0)
            #padding部分置0
            token_mask = flat_mask.unsqueeze(1).float()  # (B*H, 1, T)

        frame_mask = F.interpolate(token_mask, size=feats.shape[1], mode='nearest')  # (B*H, 1, F)
        frame_mask = frame_mask.squeeze(1).bool()  # (B*H, F)
        feats = feats * frame_mask.unsqueeze(-1).type_as(feats)

        # —— 3) Mean‑pooling (mask pad) → emb_utts (B*H, d_model)
        lengths = frame_mask.sum(dim=-1, keepdim=True)    #(B*H, 1)    #flat_mask
        lengths = lengths.clamp_min(1.0)
        emb_utts = feats.sum(dim=1) / lengths    #(B*H, d_model)

        # —— 4) Reshape 回 (B, H, d_model)
        hist_emb = emb_utts.reshape(B, H, self.d_model)

        # —— 5) GRU 聚合 → gru_out (B, H, gru_hidden)
        gru_out, _ = self.gru(hist_emb)

        # —— 6) 计算每条历史 utterance 的注意力得分
        #      mask 掉那些全为 pad 的 utterance
        utt_mask = (batch_wav_masks.sum(dim=-1) > 0)      # (B, H)
        scores   = self.attn(gru_out).squeeze(-1)        # (B, H)
        # 【修复 1：防 NaN】
        # 不要使用 float('-inf')，改用 -1e9。
        # 原因：Softmax(全-inf) = NaN，但 Softmax(全-1e9) = 均匀分布(1/H)，不会崩。
        scores   = scores.masked_fill(~utt_mask, -1e9)
        weights  = self.softmax(scores)                   # (B, H)

        # —— 7) 加权求和 → context (B, gru_hidden)
        context  = (gru_out * weights.unsqueeze(-1)).sum(dim=1)

        # —— 8) 线性映射到 out_dim → H_coarse (B, out_dim)
        H_coarse = self.linear(context)

        # 【修复 2：逻辑校正】
        # 判断哪些样本是“全空”的 (即 Turn=0，没有任何历史)
        # valid_rows 形状 (B,)，True 表示至少有一条历史，False 表示完全没历史
        valid_rows = utt_mask.any(dim=-1)

        # 如果某一行完全没历史，强制把它的 context 乘 0
        # 这一步去除了 GRU Bias 和 Softmax 均匀分布带来的噪音
        H_coarse = H_coarse * valid_rows.unsqueeze(-1).type_as(H_coarse)
        return H_coarse

# ====================================================
# 3. 音频细粒度模块 AcousticFineGrainedModule
#    supports batch_history_wavs: (B, H, T)
# ====================================================
class AcousticFineGrainedModule(nn.Module):
    def __init__(self,
                 wav2vec_model_name: str= "xxx/wav2vec2-base-100h",  #facebook/wav2vec2-base-960h
                 d_model:            int=256,
                 n_head:             int=4,
                 conv_kernel:        int=3,
                 max_history:        int = 10):
        """
        初始化 AcousticFineGrainedModule。

        参数:
        - wav2vec_model_name (str):     HuggingFace 预训练 Wav2Vec2 模型名（例如 "facebook/wav2vec2-base-960h"）。
        - d_model (int):                Wav2Vec2 输出维度（base 模型默认为 768）。
        - n_head (int):                 多头注意力中的头数。
        - conv_kernel (int):            一维卷积核大小，用于局部帧级融合。
        - max_history (int, default=10): 最大历史 utterance 数量。
        """
        super().__init__()
        # 记录最大历史 utterance 数量
        self.max_history = max_history
        self.d_model = d_model

        # 1) Wav2Vec2 模型，用于从原始波形中提取帧级特征
        self.wav2vec    = Wav2Vec2Model.from_pretrained(
            wav2vec_model_name
        )  #.to(device)
        # 冻结所有参数
        for p in self.wav2vec.parameters():
            p.requires_grad = False

        # 2) 说话人投影，将说话人 embedding 映射到 d_model 维度
        self.spk_proj   = nn.Linear(d_model, d_model)

        #因为wav2vec.config.hidden_size的维度是768
        self.feat_proj = nn.Linear(768, d_model)

        # 4) 1D 卷积，用于跨时间步融合邻近帧
        #    输入/输出通道均为 d_model，padding 保持长度不变
        self.conv       = ConvNorm(d_model, d_model,
                                   kernel_size=conv_kernel,
                                   stride=1,
                                   padding=(conv_kernel-1)//2)

        # 5) Cross‐attention，用于对齐 TTS 查询与帧级特征
        self.cross_attn = MultiHeadAttention(n_head, d_model,
                                             d_k=d_model//n_head,
                                             d_v=d_model//n_head)

        # 6) 前馈网络，内层维度为 4*d_model，使用卷积操作
        self.ffn        = PositionwiseFeedForward(d_model,
                                                  d_model*4,
                                                  kernel_size=[conv_kernel, conv_kernel])

    def forward(self,
                batch_history_wavs: torch.Tensor,
                batch_wav_masks: torch.Tensor,
                tts_query:          torch.Tensor,
                speaker_emb:        torch.Tensor,
                max_hist_frames: int,
                id=1,) -> torch.Tensor:
        """
        前向计算：对历史音频帧和 TTS 查询做细粒度融合。

        输入:
        - batch_history_wavs (Tensor): 形状 (B, H, T)
          B = batch size
          H = 历史 utterance 数量 (<= max_history)为10
          T = 每条 utterance 的采样长度（已 pad/truncate）,根据采样频率得到的音频总共采样的个数

        - tts_query (Tensor):          形状 (B, L, d_model)
          L = TTS 解码时的查询长度（输出序列长度）

        - speaker_emb (Tensor):        形状 (B, 10,d_model)
          说话人 embedding，用于条件化音频特征

        - max_hist_frames (int):        最大历史帧数（用于位置编码的长度）。

        返回:
        - out (Tensor):                形状 (B, L, d_model)
          融合后的细粒度特征，可直接送入后续解码器
        """
        device = next(self.parameters()).device
        # --- 强制把 numpy 转成 torch.Tensor 并放到同一个 device 上 ---
        if not isinstance(batch_history_wavs, torch.Tensor):
            batch_history_wavs = torch.from_numpy(batch_history_wavs).float().to(device)
        if not isinstance(batch_wav_masks, torch.Tensor):
            batch_wav_masks = torch.from_numpy(batch_wav_masks).float().to(device)

        # B: batch size, H: 历史 utterance 数, T: 每条的帧长度
        B, H, T = batch_history_wavs.shape

        # —— 1) 拍平历史 utterance和mask: (B*H, T)
        flat_wavs = batch_history_wavs.reshape(B*H, T)
        flat_mask = batch_wav_masks.reshape(B * H, T)

        try:
            save_dir = os.path.join("xxx", str(id))
            save_path = os.path.join(save_dir, "wav2vec_features.pt")
            feats=torch.load(save_path)
        except FileNotFoundError:
            with torch.no_grad():
                feats = self.wav2vec(flat_wavs, attention_mask=flat_mask)[0]
            # 把 Wav2Vec2 可能在 padding 处生产的 NaN 全关掉
            feats = torch.nan_to_num(feats, nan=0.0)


        feats=self.feat_proj(feats)
        frame_len = feats.size(1)  # F

        # —— 3) reshape → 汇集所有帧: all_feats (B, H*F, d_model)
        all_feats = feats.reshape(B, H * feats.size(1), -1)

        # —— 4) 加上说话人条件与位置编码
        # spk: (B, 10, d_model)
        spk_flat =speaker_emb.reshape(B * H, -1)  # (B*H, d_model)
        spk_proj_flat = self.spk_proj(spk_flat)  # (B*H, d_model)
        spk_proj = spk_proj_flat.unsqueeze(1).expand(-1, frame_len, -1)  # (B*H, F, d_model)
        spk_all = spk_proj.reshape(B, H * frame_len, -1)  # (B, H*F, d_model)

        # —— 5) 动态位置编码
        pos_emb = get_sinusoid_encoding_table(max_hist_frames, self.d_model).to(all_feats.device)
        pos = pos_emb[:frame_len].unsqueeze(0).expand(B * H, -1, -1)  # (B*H, F, d_model)
        pos=pos.reshape(B,H*frame_len,-1)     #(B, H*F, d_model)
        x   = all_feats + spk_all + pos


        #  paddding部分置为0，flat_mask: (B*H, T) 这里需要拿回 flat_mask
        token_mask = flat_mask.unsqueeze(1).float()  # (B*H, 1, T)
        frame_mask = F.interpolate(token_mask, size=feats.shape[1], mode='nearest')  # (B*H, 1, F)
        frame_mask = frame_mask.squeeze(1)  # (B*H, F)


        # 把它 reshape 回 (B, H*F)
        frame_mask = frame_mask.reshape(B, H * frame_len)  # (B, H*F)
        # 把 x 中对应 padding 的帧向量都置为 0
        x = x * frame_mask.unsqueeze(-1).type_as(x)  # (B, H*F, d_model)

        # —— 6) 1D 卷积融合临近帧: 需在通道维度做卷积
        x = self.conv(x.transpose(1,2)).transpose(1,2)
        # x.shape 保持 (B, H*F, d_model)

        # 再次 Mask 清洗 (消除卷积 Bias 对 Padding 的影响)
        x = x * frame_mask.unsqueeze(-1).type_as(x)

        # —— 7) Cross‐Attention 对齐 tts_query 与帧特征
        #    key/value = x, query = tts_query
        attn_mask = (frame_mask==0).unsqueeze(1).to(torch.bool)  # 扩展为 (B, 1, H*F)

        attn_mask = self.fix_attn_mask(attn_mask, Lq=tts_query.size(1))
        attn_out, _ = self.cross_attn(tts_query, x, x, mask=attn_mask)
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)

        # —— 8) 前馈网络 + 残差，输出 (B, L, d_model)
        out = self.ffn(attn_out + tts_query)

        #【优化】强制清洗Turn = 0的样本
        #    valid_rows: (B,) True 表示至少有一帧是真实的
        valid_rows = (frame_mask.sum(dim=-1) > 0).float()
        out = out * valid_rows.view(B, 1, 1).type_as(out)
        return out

    def fix_attn_mask(self,attn_mask: torch.Tensor, Lq: int) -> torch.Tensor:
        """
        修复 attention mask，确保每个 query 至少能 attend 到一个 key。

        参数:
        - attn_mask: BoolTensor, 形状 (B, 1, S)，True 表示被屏蔽（padding）
        - Lq: int, query 序列长度

        返回:
        - new_mask: BoolTensor, 形状 (B, 1, S)，修正后的 mask
        """
        B, _, S = attn_mask.shape
        # 1) squeeze -> (B, S)
        base_mask = attn_mask.squeeze(1)  # (B, S)

        # 2) expand -> (B, Lq, S)
        mask_expanded = base_mask.unsqueeze(1).expand(-1, Lq, -1).clone()  # (B, Lq, S)

        # 3) 找出哪些行全被屏蔽了
        all_masked = (mask_expanded.sum(dim=-1) == S)  # (B, Lq)；True 表示这一行全 True

        if all_masked.any():
            # 拿到所有 (i,j) 坐标
            idx = all_masked.nonzero(as_tuple=False)  # Tensor of shape (K, 2): [[i0, j0], [i1, j1], ...]
            # 把这些行的 key=0 置为 False（表示“不屏蔽”）
            mask_expanded[idx[:, 0], idx[:, 1], 0] = False

        # 4) 收回 (B,1,S)
        new_mask = mask_expanded[:, :1, :].to(attn_mask.dtype)
        return new_mask

#对于说话人id生成说话人嵌入
class SpeakerEmbeddingLayer(nn.Module):
    def __init__(self,
                 num_speakers: int = 2,   # 说话人类别数
                 emb_dim:       int = 256  # 嵌入维度
                ):
        super().__init__()
        # 普通的 Embedding 层
        self.speaker_embedding = nn.Embedding(num_speakers, emb_dim)

    def forward(self,
                speaker_ids: torch.LongTensor  # (B, T)，每个元素 ∈[0, num_speakers)
               ) -> torch.Tensor:
        """
        输入:
          speaker_ids: LongTensor of shape (B, T)
        输出:
          speaker_embs: FloatTensor of shape (B, T, emb_dim)
        """
        if not isinstance(speaker_ids, torch.Tensor):
            speaker_ids = torch.tensor(speaker_ids, dtype=torch.long, device=next(self.parameters()).device)
        # 直接调用 Embedding 就会在最后一维上扩成 embedding
        speaker_embs = self.speaker_embedding(speaker_ids)
        # speaker_embs.shape == (B, T, emb_dim)
        return speaker_embs

#特征融合模块，将两个特定特征和一个共同特征融合，门控还是普通融合？
class GatedFusion(nn.Module):
    """
    通过门控机制来融合特征。
    """
    def __init__(self, feature_dim=256, num_features=3, dropout=0.1):
        super(GatedFusion, self).__init__()

        combined_dim = feature_dim * num_features

        # 学习门控权重的网络
        self.gating_net = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.Sigmoid()
        )

        self.output_net = nn.Sequential(
            nn.Linear(combined_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, common_feature, specific_feature_text, specific_feature_audio):
        # 1. 拼接所有特征
        fused_feature = torch.cat([common_feature, specific_feature_text, specific_feature_audio], dim=1)

        # 2. 计算门控权重
        gates = self.gating_net(fused_feature)

        # 3. 应用门控
        gated_feature = fused_feature * gates

        # 4. 最终输出
        output = self.output_net(gated_feature)

        return output

#提取特定特征，实例化两次，分别提取音频特定特征和文本特定特征
class SubnetTCNSE(nn.Module):
    def __init__(self, channels=256, bottleneck=64, kernel_size=3):
        super().__init__()
        self.reduce = nn.Conv1d(channels, bottleneck, kernel_size=1)
        self.depthwise = nn.Conv1d(bottleneck, bottleneck, kernel_size=kernel_size,
                                   padding=kernel_size//2, groups=bottleneck)
        self.pointwise = nn.Conv1d(bottleneck, bottleneck, kernel_size=1)
        # Squeeze-Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(bottleneck, bottleneck//4, 1),
            nn.ReLU(inplace=False),
            nn.Conv1d(bottleneck//4, bottleneck, 1),
            nn.Sigmoid()
        )
        self.restore = nn.Conv1d(bottleneck, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        # x: [B, T, C],C=256
        b, t, c = x.size()

        # ==========================================
        # 1. 自动检测全是 0 的样本 (Auto-Detect Zero Samples)
        # ==========================================
        # 计算每个样本的绝对值总和。
        # 如果 sum == 0，说明是 Turn=0 的全空样本；如果 sum > 0，说明是有效样本。
        # shape: (B,) -> (B, 1, 1) 用于最后的广播乘法
        sample_mask = (x.abs().sum(dim=(1, 2)) > 0).float().view(b, 1, 1).type_as(x)

        x0 = x
        x = x.transpose(1,2)        # → [B, C, T]
        x = self.reduce(x)          # → [B, bottleneck, T]
        x = self.act(x)
        x = self.depthwise(x)       # → [B, bottleneck, T]
        x = self.pointwise(x)       # → [B, bottleneck, T]
        # SE
        w = self.se(x)              # → [B, bottleneck, 1]
        x = x * w                    # 通道重标定
        x = self.restore(x)         # → [B, C, T]
        x = x.transpose(1,2)        # → [B, T, C]
        # 残差 + LayerNorm + ReLU
        x = self.norm(x + x0)
        x = self.act(x)

        # ==========================================
        # 2. 最后的强制清洗 (Final Cleaning)
        # ==========================================
        # 将那些原本输入全为 0 的样本，输出强制置回 0
        x = x * sample_mask
        return x


class SequenceGLFK(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        # 1. Global: 融合两个模态 (512 -> 256)
        self.global_conv = nn.Sequential(
            nn.Conv1d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )

        # 2. Local: 捕捉时序上下文 (Depthwise Conv, k=3)
        # 保持 T 不变，padding=1
        self.local_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )

        # 3. Fine: 升维进行细粒度特征交互 (256 -> 512)
        # 对应论文中的 expanding [cite: 944]
        self.fine_conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.BatchNorm1d(dim * 2),
            nn.GELU()
        )

        # 4. Knowledge: 降维输出 (512 -> 256)
        # 对应论文中的 reducing to F_c^*
        self.knowledge_conv = nn.Conv1d(dim * 2, dim, kernel_size=1)
        self.final_act = nn.Tanh()  # 有助于解耦

    def forward(self, text_feat, audio_feat):
        # 输入: (B, T, 256)
        # ==========================================
        # 1. 自动检测 Turn=0 (全空) 样本
        # ==========================================
        # 如果 text 和 audio 的绝对值之和都是 0，说明这是个空样本
        # shape: (B, T, 256) -> sum -> (B,)
        # 只要 text 或 audio 任意一个有数据，就算有效；全为0则无效。
        has_data = (text_feat.abs().sum(dim=(1, 2)) + audio_feat.abs().sum(dim=(1, 2))) > 0

        # 制作 mask: (B, 1, 1) 用于最后的广播乘法
        valid_sample_mask = has_data.float().view(-1, 1, 1).type_as(text_feat)

        # 拼接: (B, T, 512)
        concat_feat = torch.cat([text_feat, audio_feat], dim=-1)

        # 转置以适应 Conv1d: (B, 512, T)
        x = concat_feat.transpose(1, 2)

        # Global: 融合模态
        x = self.global_conv(x)

        # Residual Connection (可选，如果 Global 后维度匹配)
        res = x

        # Local: 时序信息
        x = self.local_conv(x)
        x = x + res  # 残差连接

        # Fine -> Knowledge: 瓶颈结构提取精华
        x = self.fine_conv(x)
        x = self.knowledge_conv(x)

        # Tanh 约束
        common_feat = self.final_act(x)

        # 转置回: (B, T, 256)
        output = common_feat.transpose(1, 2)
        # ==========================================
        # 3. 最后的强制清洗 (Final Cleaning)
        # ==========================================
        # 将那些原本 Turn=0 的样本，输出强制置回 0
        output = output * valid_sample_mask

        return output


class HistoryGuidedContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        """
        基于历史情感指导的共性特征三元组损失 (Top-1 策略)

        Args:
            margin: 三元组损失的边界值 (通常设置在 0.1 ~ 0.3 之间)
        """
        super().__init__()
        self.margin = margin

    def get_teacher_indices(self, history_embeddings):
        """
        直接返回 Top-1 正样本和负样本的【索引】，不再需要生成笨重的 Mask
        """
        batch_size = history_embeddings.shape[0]

        # 1. 历史向量归一化 & 计算自相似度矩阵
        hist_norm = F.normalize(history_embeddings, p=2, dim=1)
        teacher_sim = torch.matmul(hist_norm, hist_norm.T)

        # 2. 屏蔽对角线 (自己不能做自己的正/负样本)
        identity = torch.eye(batch_size, device=history_embeddings.device).bool()

        # 找正样本：对角线设为 -inf，这样选 argmax 时绝对不会选到自己
        sim_for_pos = teacher_sim.clone()
        sim_for_pos.masked_fill_(identity, -float('inf'))

        # 找负样本：对角线设为 inf，这样选 argmin 时绝对不会选到自己
        sim_for_neg = teacher_sim.clone()
        sim_for_neg.masked_fill_(identity, float('inf'))

        # 3. 找出 Top-1 正样本和负样本的索引
        top1_pos_idx = sim_for_pos.argmax(dim=1)
        top1_neg_idx = sim_for_neg.argmin(dim=1)

        return top1_pos_idx, top1_neg_idx

    def forward(self, common_features, history_embeddings):
        """
        Args:
            common_features: (B, 256) -> 学生模型的特征 (Anchor)
            history_embeddings: (B, 1024) -> 老师(历史)模型的特征，用来当裁判
        """
        # =========================================================
        # 1. 提取三元组 (Anchor, Positive, Negative)
        # =========================================================
        with torch.no_grad():
            pos_idx, neg_idx = self.get_teacher_indices(history_embeddings)

        anchor = common_features
        positive = common_features[pos_idx]
        negative = common_features[neg_idx]

        # =========================================================
        # 2. 计算 Triplet Margin Loss
        # =========================================================
        # 语音/NLP领域常用的余弦相似度 Triplet Loss
        # 公式: L = max(0, sim(Anchor, Negative) - sim(Anchor, Positive) + margin)

        anchor_norm = F.normalize(anchor, p=2, dim=1)
        pos_norm = F.normalize(positive, p=2, dim=1)
        neg_norm = F.normalize(negative, p=2, dim=1)

        # 计算对应位置的点积（即余弦相似度），结果 shape 为 (B,)
        sim_ap = (anchor_norm * pos_norm).sum(dim=1)
        sim_an = (anchor_norm * neg_norm).sum(dim=1)

        # 期望 sim_ap 尽量大，sim_an 尽量小，且差距至少为 margin
        losses = F.relu(sim_an - sim_ap + self.margin)

        return losses.mean()


class RandomContrastiveLoss(nn.Module):
    def __init__(self, margin=0.3):
        """
        基于纯随机采样的三元组补充实验 Loss

        Args:
            margin: Triplet Loss 的边界值 (推荐 0.2 ~ 0.5，默认 0.3)
        """
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, common_features, history_embeddings=None):
        """
        Args:
            common_features: (B, 256) -> 模型提取出的共性特征
            history_embeddings: (B, 1024) -> 占位，不参与计算
        """
        batch_size = common_features.shape[0]
        device = common_features.device

        if batch_size < 3:
            return torch.tensor(0.0, requires_grad=True, device=device)

        # =========================================================
        # 0. 极其关键的一步：特征 L2 归一化
        # =========================================================
        # 只有将特征映射到单位超球面上，margin=0.3 才具有稳定的数学意义
        common_features = F.normalize(common_features, p=2, dim=1)

        p_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        n_indices = torch.zeros(batch_size, dtype=torch.long, device=device)

        # =========================================================
        # 1. 随机分配正负样本
        # =========================================================
        for i in range(batch_size):
            pool = torch.tensor([j for j in range(batch_size) if j != i], device=device)
            rand_idx = torch.randperm(pool.size(0))
            p_indices[i] = pool[rand_idx[0]]
            n_indices[i] = pool[rand_idx[1]]

        # =========================================================
        # 2. 提取并计算 Loss
        # =========================================================
        anchors = common_features
        positives = common_features[p_indices]
        negatives = common_features[n_indices]

        loss = self.triplet_loss(anchors, positives, negatives)

        return loss



#提取共同特征中的FC模块
class FCFeatureExtractor(nn.Module):
    """
    一个简单的全连接 (FC) 模块：
    输入     -> (B, T, 256)
      先在最后一个维度上做线性映射：256 → 256
      再加上 ReLU 激活
    输出     -> (B, T, 256)
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        # 在 “通道/特征” 维度上做全连接：256 → 256
        self.fc = nn.Linear(feat_dim, feat_dim, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: torch.Tensor of shape (B, T, 256)
        Returns:
          out: torch.Tensor of shape (B, T, 256)
        """
        # 直接把 nn.Linear 应用于最后一个维度即可；PyTorch 会自动对 (B,T,256) 中的每个时刻 t 应用 fc
        out = self.fc(x)    # → (B, T, 256)
        out = self.relu(out)
        return out


###用于重构损失的解码器,文本和音频可以复用，这里只是命名的时候用文本
class ReconstructionModule_text(nn.Module):
    """
    一个独立的模块，用于从共性特征和特定特征中重构原始模态特征。

    它接收分解后的特征，将它们拼接起来，通过一个解码器（1D卷积层）
    来重构原始特征，并计算重构损失。
    """

    def __init__(self, feature_dim: int=256):
        """
        初始化重构模块。

        参数:
            feature_dim (int): 输入及输出特征的维度 (即代码中的 C)。
                               在你的案例中，这个值是 256。
        """
        super(ReconstructionModule_text, self).__init__()

        # 定义解码器。
        # 输入通道数是 feature_dim * 2，因为我们将共性特征和特定特征拼接在了一起。
        # 输出通道数是 feature_dim，即我们希望恢复的原始特征维度。
        # 使用 1x1 卷积核，它在功能上等同于一个作用于特征维度上的全连接层，
        # 非常适合在保持序列长度不变的情况下进行通道变换。
        self.decoder = nn.Conv1d(in_channels=feature_dim * 2,
                                 out_channels=feature_dim,
                                 kernel_size=1,
                                 bias=False)

        # 定义用于计算重构误差的损失函数
        self.loss_function = nn.MSELoss()

    def forward(self,
                common_feature: torch.Tensor,
                specific_feature: torch.Tensor,
                original_feature_to_reconstruct: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        执行前向传播，完成特征的重构和损失计算。

        参数:
            common_feature (torch.Tensor): 共性特征张量，shape: (B, T, C)
            specific_feature (torch.Tensor): 特定特征张量，shape: (B, T, C)
            original_feature_to_reconstruct (torch.Tensor): 用于计算损失的目标原始特征，shape: (B, T, C)

        返回:
            tuple[torch.Tensor, torch.Tensor]: 包含两个元素的元组
                - reconstructed_feature (torch.Tensor): 重构出的特征，shape: (B, T, C)
                - reconstruction_loss (torch.Tensor): 计算出的重构损失（标量）
        """
        # 1. 预处理：将输入从 (B, T, C) 转换为 (B, C, T) 以适配 Conv1d
        common_feature_t = common_feature.permute(0, 2, 1)    #(B,256,t)
        specific_feature_t = specific_feature.permute(0, 2, 1)
        original_feature_t = original_feature_to_reconstruct.permute(0, 2, 1)

        # 2. 拼接：沿特征维度 (dim=1) 拼接共性与特定特征
        combined_feature = torch.cat([common_feature_t, specific_feature_t], dim=1)    #(B,512,t)

        # 3. 解码：通过解码器进行重构
        reconstructed_feature_t = self.decoder(combined_feature)

        # 4. 计算损失：计算重构特征与原始特征之间的MSE损失
        reconstruction_loss_text = self.loss_function(reconstructed_feature_t, original_feature_t)

        # 5. 后处理：将输出变回 (B, T, C) 格式，方便后续使用
        reconstructed_feature_text = reconstructed_feature_t.permute(0, 2, 1)

        return reconstructed_feature_text, reconstruction_loss_text



#####模态特定重构
class SpecificReconstructionModule(nn.Module):
    """
    一个独立的模块，用于实现模态特定的重构过程。

    它接收由前一阶段解码器重构出的特征 (X'_m)，
    并将其送入模态特定的编码器 (E_sp_m)，以得到再次估计出的特定特征 (Sp_m')。
    最后，计算 Sp_m' 与原始特定特征 Sp_m 之间的损失 L_s。
    """

    def __init__(self, specific_encoder: SubnetTCNSE):
        """
        初始化模块。

        参数:
            specific_encoder (SubnetTCNSE): 一个已经实例化的、来自主模型的
                                          模态特定编码器。本模块将复用此编码器。
        """
        super(SpecificReconstructionModule, self).__init__()

        # 直接引用传入的、已存在的编码器实例
        self.specific_encoder = specific_encoder

        # 定义用于计算特定重构误差的损失函数
        self.loss_function = nn.MSELoss()

    def forward(self,
                reconstructed_feature: torch.Tensor,
                original_specific_feature: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        执行前向传播。

        参数:
            reconstructed_feature (torch.Tensor): 由通用解码器D_m重构出的特征 X'_m。
                                                  期望 shape: (B, T, 256)。
            original_specific_feature (torch.Tensor): 原始的模态特定特征 Sp_m。
                                                       期望 shape: (B, T, 256)。

        返回:
            一个元组，包含:
                - re_estimated_specific_feature (torch.Tensor): 再次估计出的特定特征 Sp_m'，
                  shape: (B, T, 256)。
                - specific_reconstruction_loss (torch.Tensor): 计算出的标量损失 L_s。
        """
        # 你的 SubnetTCNSE 编码器期望的输入 shape 是 (B, T, C)，
        # 所以我们在这里不需要进行维度转换。

        # 1. 再次编码：将重构后的特征传入特定的编码器。
        #    这对应论文中的公式 (5): Sp_m' = E_sp_m(X'_m)
        re_estimated_specific_feature = self.specific_encoder(reconstructed_feature)

        # 2. 计算损失：计算再次估计出的特征和原始特定特征之间的均方误差。
        #    这对应论文中的公式 (6): L_s = ||Sp^m - Sp_m'||^2
        specific_reconstruction_loss = self.loss_function(
            re_estimated_specific_feature,
            original_specific_feature
        )

        return re_estimated_specific_feature, specific_reconstruction_loss


#####软正交损失
class OrthogonalityLossModule(nn.Module):
    """
    修正版：真正的正交性损失。
    目标：最小化两个向量序列之间余弦相似度的【平方】。
    无论正相关(1)还是负相关(-1)，Loss 都最大；只有垂直(0)时，Loss 最小。
    """
    def __init__(self):
        super(OrthogonalityLossModule, self).__init__()
        # 不需要内部 Loss 函数，直接手算更准确

    def forward(self,
                common_feature: torch.Tensor,
                specific_feature_text: torch.Tensor,
                specific_feature_audio: torch.Tensor) -> torch.Tensor:
        """
        输入: (B, T, C)
        """
        # 1. 归一化 (L2 Normalize)
        # 沿着特征维度 (dim=2) 做归一化，这样点积就是余弦相似度
        common_norm = F.normalize(common_feature, p=2, dim=2)
        text_norm   = F.normalize(specific_feature_text, p=2, dim=2)
        audio_norm  = F.normalize(specific_feature_audio, p=2, dim=2)

        # 2. 计算余弦相似度 (Cosine Similarity)
        # (B, T, C) * (B, T, C) -> (B, T, C) -> sum(dim=2) -> (B, T)
        # 对应位置点积
        cos_sim_text = (common_norm * text_norm).sum(dim=2)
        cos_sim_audio = (common_norm * audio_norm).sum(dim=2)

        # 3. 计算损失：余弦相似度的平方 (Squared Cosine Similarity)
        # 这样 -1 和 1 都会产生最大的 Loss (1)，只有 0 产生 Loss 0
        loss_ortho_text = (cos_sim_text ** 2).mean()
        loss_ortho_audio = (cos_sim_audio ** 2).mean()

        # 4. 汇总
        total_orthogonality_loss = loss_ortho_text + loss_ortho_audio

        return total_orthogonality_loss


####使用兰姆达超参数组合成总的解耦损失，解耦损失这里的超参数可以调整
def calculate_decoupling_loss(loss_r, loss_s, loss_m, loss_o, lambda_r=1, lambda_s=0.5, lambda_m=1, lambda_o=20):
    """
    根据论文公式 (9) 计算总的解耦损失 Ld。

    参数:
        loss_r (torch.Tensor): 重构损失 (L_r)。
        loss_s (torch.Tensor): 特定重构损失 (L_s)。
        loss_m (torch.Tensor): 三元组损失 (L_m)。
        loss_o (torch.Tensor): 正交性损失 (L_o)。
        lambda_r (float): L_r 的权重系数。 0.1
        lambda_s (float): L_s 的权重系数。 0.1
        lambda_m (float): L_m 的权重系数。 0.01
        lambda_o (float): L_o 的权重系数。 0.01

    返回:
        torch.Tensor: 加权求和后的总解耦损失 Ld。
    """

    decoupling_loss = (lambda_r * loss_r) + \
                      (lambda_s * loss_s) + \
                      (lambda_m * loss_m) + \
                      (lambda_o * loss_o)

    return decoupling_loss



#注意力的核心代码
class MultiheadAttention_LFA(nn.Module):
    """ Multi-Head Attention (LFA专用版本) """

    def __init__(self, embed_dim, num_heads, attn_dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, attn_mask=None):
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()
        tgt_len, bsz, embed_dim = query.size()
        if qkv_same:
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        elif kv_same:
            q = F.linear(query, self.in_proj_weight[:embed_dim, :],
                         self.in_proj_bias[:embed_dim] if self.in_proj_bias is not None else None)
            k, v = F.linear(key, self.in_proj_weight[embed_dim:, :],
                            self.in_proj_bias[embed_dim:] if self.in_proj_bias is not None else None).chunk(2, dim=-1)
        else:
            q = F.linear(query, self.in_proj_weight[:embed_dim, :],
                         self.in_proj_bias[:embed_dim] if self.in_proj_bias is not None else None)
            k = F.linear(key, self.in_proj_weight[embed_dim:2 * embed_dim, :],
                         self.in_proj_bias[embed_dim:2 * embed_dim] if self.in_proj_bias is not None else None)
            v = F.linear(value, self.in_proj_weight[2 * embed_dim:, :],
                         self.in_proj_bias[2 * embed_dim:] if self.in_proj_bias is not None else None)
        q = q * self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        return attn, attn_weights

######位置编码相关
class PositionalEncoding(nn.Module):
    def __init__(self, d_hid=256, n_position=1500):
        super(PositionalEncoding, self).__init__()
        # register_buffer 的作用是创建一个模型参数，但这个参数不会被 optimizer 更新
        # 这很适合位置编码，因为它是固定的
        self.register_buffer('pos_table', get_sinusoid_encoding_table(n_position, d_hid, padding_idx=0))

    def forward(self, x):
        """
        x: 输入张量，shape: (T, B, C)
        """
        # 从位置编码表中取出与序列长度 T 对应的部分，并与输入 x 相加
        # self.pos_table[:x.size(0), :] 的 shape 是 (T, C)
        # .clone().detach() 确保这部分操作不会影响反向传播
        return x + self.pos_table[:x.size(0), :].clone().detach()

###包装了多头注意力+前馈神经网络（FNN）
class TransformerEncoderLayer_LFA(nn.Module):
    """ Transformer Encoder Layer (LFA专用版本) """

    def __init__(self, embed_dim=256, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 内部调用也使用重命名后的版本
        self.self_attn = MultiheadAttention_LFA(embed_dim=self.embed_dim, num_heads=self.num_heads,
                                                attn_dropout=attn_dropout)
        self.attn_mask = attn_mask
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True
        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        residual = x
        ##前置层归一化，将层归一化放在了计算之前，这种结构通常能让模型训练过程更稳定。
        x = self.maybe_layer_norm(0, x, before=True)
        if x_k is None and x_v is None:
            ###自注意力
            x, _ = self.self_attn(query=x, key=x, value=x)
        else:
            ##交叉注意力
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v)

        ##dropout加残差连接
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x

        ##继续层归一化
        x = self.maybe_layer_norm(0, x, after=True)
        residual = x
        ###FNN，FNN 通常特指一个由两个全连接层和它们之间的一个非线性激活函数（通常是 ReLU）组成的特定结构。
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    #层归一化
    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerEncoder_LFA(nn.Module):
    """ Transformer Encoder (LFA专用版本) """

    #init四个参数256，8，2，0.1
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        # 内部调用也使用重命名后的版本
        self.layers = ModuleList(
            [TransformerEncoderLayer_LFA(embed_dim, num_heads, attn_dropout, relu_dropout, res_dropout, attn_mask) for _
             in range(layers)])
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        x = self.embed_scale * x_in   #乘以缩放因子
        x = F.dropout(x, p=self.dropout, training=self.training)   #应用dropout
        ##如果k，v非空，对k，v也要缩放并应用dropout
        if x_in_k is not None and x_in_v is not None:
            x_k, x_v = self.embed_scale * x_in_k, self.embed_scale * x_in_v
            x_k, x_v = F.dropout(x_k, p=self.dropout, training=self.training), F.dropout(x_v, p=self.dropout,
                                                                                         training=self.training)

        ###layers是两层的TransformerEncoderLayer_LFA（每一层都有一个自注意力或交叉注意力）
        for layer in self.layers:
            #走交叉注意力
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            #走自注意力
            else:
                x = layer(x)

        ####又一个归一化？
        if self.normalize:
            x = self.layer_norm(x)
        return x


# ==============================================================================
# 2. LFA 模块现在调用这些重命名后的、无冲突的组件
# ==============================================================================
####不知道是文本好还是音频好
class LanguageFocusedAttractor(nn.Module):
    """
    语言专注吸引子 (LFA) 模块的双模态适配版。
    """

    def __init__(self, feature_dim=256, n_heads=8, n_layers=2, dropout=0.1):
        super(LanguageFocusedAttractor, self).__init__()

        # 分支1: 文本自注意力 (L -> L)
        self.trans_l_self = TransformerEncoder_LFA(
            embed_dim=feature_dim,
            num_heads=n_heads,
            layers=n_layers,
            attn_dropout=dropout,
            # ... 其他参数
        )

        # 分支2: 音频到文本的交叉注意力 (A -> L)
        self.trans_l_with_a = TransformerEncoder_LFA(
            embed_dim=feature_dim,
            num_heads=n_heads,
            layers=n_layers,
            attn_dropout=dropout,
            # ... 其他参数
        )

    def forward(self, specific_feature_text: torch.Tensor, specific_feature_audio: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
                执行前向传播，返回两个独立的增强特征。

                参数:
                    specific_feature_text (torch.Tensor): 文本的特定特征 Sp_l，shape: (B, T_text, C)。
                    specific_feature_audio (torch.Tensor): 音频的特定特征 Sp_a，shape: (B, T_audio, C)。

                返回:
                    一个元组，包含两个张量:
                    - enhanced_text_from_self (torch.Tensor): 经过自注意力增强的文本特征，shape: (B, C)。
                    - enhanced_text_from_audio (torch.Tensor): 吸收了音频信息后增强的文本特征，shape: (B, C)。
                """
        ##将两个张量的维度从 (B, T, 256) 转换成 (T, B, 256)。这是因为很多 Transformer 的实现（包括这里的 TransformerEncoder_LFA）都期望序列长度维度在前。
        sp_l = specific_feature_text.permute(1, 0, 2)
        sp_a = specific_feature_audio.permute(1, 0, 2)

        #enhanced_l_from_l = self.trans_l_self(sp_l)
        enhanced_l_from_a = self.trans_l_self(sp_a)
        #enhanced_l_from_a = self.trans_l_with_a(sp_l, sp_a, sp_a)
        enhanced_l_from_l = self.trans_l_with_a(sp_a, sp_l, sp_l)

        last_hidden_l = enhanced_l_from_l[-1]   #取最后一个序列，shape为（B,256）
        last_hidden_a = enhanced_l_from_a[-1]  #同理

        return last_hidden_l,last_hidden_a

#####对共享特征进行全局摘要（B,T,256）-》（B,256）
class SharedFeatureProcessor(nn.Module):
    """
    一个独立的模块，用于处理共享特征序列。

    该模块严格遵循 DLF 论文中的设计：
    1. 使用一个独立的 Transformer Encoder 对输入的共享特征序列进行自注意力计算，
       以捕捉序列内部的上下文关系。
    2. 提取序列的最后一个时间步的隐藏状态作为全局摘要。
    3. 将该摘要向量通过一个包含两层全连接（FC）网络和残差连接的模块进行增强。
    """

    def __init__(self, feature_dim=256, n_heads=8, n_layers=2, dropout=0.1):
        """
        初始化模块。

        参数:
            feature_dim (int): 输入特征的维度 (C)。
            n_heads (int): Transformer中多头注意力的头数。
            n_layers (int): Transformer的层数。
            dropout (float): Dropout概率。
        """
        super(SharedFeatureProcessor, self).__init__()

        # 1. Transformer (Self-attention) 部分
        #    这个编码器专门用于处理共享特征
        self.self_attention_encoder = TransformerEncoder_LFA(
            embed_dim=feature_dim,
            num_heads=n_heads,
            layers=n_layers,
            attn_dropout=dropout,
            relu_dropout=dropout,
            res_dropout=dropout,
            embed_dropout=dropout
        )

        # 2. 2 FC Layers + Projector 部分
        #    在原论文代码中，这部分通常是一个带残差连接的前馈网络
        self.projection_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, common_feature_sequence: torch.Tensor) -> torch.Tensor:
        """
        执行前向传播。

        参数:
            common_feature_sequence (torch.Tensor): 共享特征序列 Sh，shape: (B, T, 256)。

        返回:
            torch.Tensor: 经过处理和摘要后的全局共享特征 HSh，shape: (B, 256)。
        """
        # 1. 预处理：将输入从 (B, T, C) 转换为 (T, B, C) 以适配Transformer
        common_feature_t = common_feature_sequence.permute(1, 0, 2)

        # 2. 通过 Transformer Encoder 进行自注意力计算
        enhanced_sequence_t = self.self_attention_encoder(common_feature_t)

        # 3. 序列摘要：提取最后一个时间步的隐藏状态作为全局特征
        global_feature = enhanced_sequence_t[-1]  # shape: (B, 256)

        # 4. 通过前馈网络进行最终的增强和投射
        #    首先保存一份用于残差连接
        residual = global_feature

        #    通过网络
        projected_feature = self.projection_net(global_feature)

        #    添加残差连接并进行层归一化
        final_feature = self.layer_norm(residual + projected_feature)

        return final_feature
