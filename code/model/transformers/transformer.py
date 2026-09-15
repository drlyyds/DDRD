import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from text.symbols import symbols

from .constants import PAD
from .blocks import (
    get_sinusoid_encoding_table,
    LinearNorm,
)


class TextEncoder(nn.Module):
    """ Text Encoder，输入src_seq，mask(布尔/二值张量)，形状都为(Batchsize,T),T为当前batch中音素序列的最大长度，(因为不满足的在dataset中会被扩充)\n
                     输出enc_output(最终的上下文感知向量，形状也是B,T,256), src_word_emb形状为(Batchsize,T，256),它是离散音素id序列映射的嵌入向量(没有经过transformer的上下文感知)"""

    def __init__(self, config):
        super(TextEncoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        """n_src_vocab就相当于我预设的音素字典(包括标点符号，特殊符号等)，这样我对每个音频的音素序列中的东西，都能分配一个独立的向量 ID。"""
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=PAD
        )  #这一行代码在做的是“给离散的符号 ID（例如你前面定义好的 symbols 里的每个字符／音素）映射成一个连续的向量”，它等价于一个查表操作，你可以把它想象成一个大小为 (n_src_vocab × d_word_vec) 的矩阵
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        src_word_emb = self.src_word_emb(src_seq)
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = src_word_emb + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = src_word_emb + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, src_word_emb


class Decoder(nn.Module):
    """ 输入形状和输出形状一致，输出形状为(B,D，256)，这个可以理解为音频的梅尔频谱的初始预测，然后经过mel_linear，变为(B,D，80)和postnet得到最终的梅尔频谱 """

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]  #256
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        """1500"""
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask


class FFTBlock(nn.Module):
    """ FFT Block """

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, layer_norm=True):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = LinearNorm(d_model, n_head * d_k)
        self.w_ks = LinearNorm(d_model, n_head * d_k)
        self.w_vs = LinearNorm(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.sqrt(d_k))
        self.layer_norm = nn.LayerNorm(d_model) if layer_norm else None

        self.fc = LinearNorm(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q/k/v: (B, L, d_model), mask: (B,1,Lk) or (B*n_head, Lq, Lk)
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape

        residual = q

        # 1) 线性映射并拆头
        q = self.w_qs(q).view(B, Lq, self.n_head, self.d_k)
        k = self.w_ks(k).view(B, Lk, self.n_head, self.d_k)
        v = self.w_vs(v).view(B, Lk, self.n_head, self.d_v)

        # 2) 合并 batch 与 head 维度，得到 (B*n_head, L, d_k/v)
        q = q.permute(2, 0, 1, 3).reshape(-1, Lq, self.d_k)
        k = k.permute(2, 0, 1, 3).reshape(-1, Lk, self.d_k)
        v = v.permute(2, 0, 1, 3).reshape(-1, Lk, self.d_v)

        # 3) 处理 mask：扩成 (B*n_head, Lq, Lk)，并确保 True 表示要 **屏蔽** 的位置
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1)

        # 4) attention
        output, attn = self.attention(q, k, v, mask=mask)

        # 5) 恢复形状到 (B, Lq, n_head * d_v)
        output = output.view(self.n_head, B, Lq, self.d_v)
        output = (output.permute(1, 2, 0, 3)
                        .contiguous()
                        .view(B, Lq, -1))

        # 6) 最后的全连接、残差
        output = self.dropout(self.fc(output))
        output = output + residual

        # —— 在进入 layer_norm 之前，清掉所有 NaN/Inf ——
        output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

        if self.layer_norm is not None:
            output = self.layer_norm(output)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # q: (B*n_head, Lq, d_k), k,v: (B*n_head, Lk/Lv, d_k/d_v)
        attn = torch.bmm(q, k.transpose(1, 2))  # (B*n, Lq, Lk)
        attn = attn / self.temperature

        if mask is not None:
            # mask: same shape as attn, True for **padding** positions
            attn = attn.masked_fill(mask, float('-inf'))

        attn = torch.clamp(attn, max=1e4)  # 可选，一般用 clamp 也能防溢出
        attn = torch.nan_to_num(attn, neginf=float('-inf'))
        # 1) softmax
        weights = self.softmax(attn)
        # 2) 再清一遍 NaN/Inf
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        output = torch.bmm(weights, v)
        return output, weights


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1, layer_norm=True):
        super(PositionwiseFeedForward, self).__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in) if layer_norm else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = output + residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)

        return output
