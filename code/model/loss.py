import torch
import torch.nn as nn
import torch.nn.functional as F
from text import sil_phonemes_ids


class CompTransTTSLoss(nn.Module):
    """ CompTransTTS Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(CompTransTTSLoss, self).__init__()
        self.loss_config = train_config["loss"]
        """loss_config是一个字典"""
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        """"pitch_feature_level=phoneme_level"""
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        """"energy_feature_level=phoneme_level"""
        self.learn_alignment = model_config["duration_modeling"]["learn_alignment"]
        """learn_alignment=true"""
        self.binarization_loss_enable_steps = train_config["duration"]["binarization_loss_enable_steps"]
        """binarization_loss_enable_steps=18000"""
        self.binarization_loss_warmup_steps = train_config["duration"]["binarization_loss_warmup_steps"]
        """binarization_loss_warmup_steps=10000"""
        self.sum_loss = ForwardSumLoss()
        """sum_loss是自定义的ForwardSumLoss"""
        self.bin_loss = BinLoss()
        """bin_loss是自定义的BinLoss"""
        self.mse_loss = nn.MSELoss()
        """mse_loss是MSELoss"""
        self.mae_loss = nn.L1Loss()
        """mae_loss是L1Loss"""
        self.sil_ph_ids = sil_phonemes_ids()
        """sil_ph_ids是一个list，list里面的是静音符号的id"""
        self.var_start_steps = train_config["step"]["var_start_steps"]
        """val_step=10000"""

    #losses = Loss(batch, output, step=step) #output就是(B,D,80)
    def forward(self, inputs, predictions, step,decoupling_loss):
        (
            texts,
            _,
            _,
            mel_targets,  #(B,D,80)
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            *_,
        ) = inputs[3:]
        #inputs[3:]为(texts,text_lens,max(text_lens),mels,mel_lens,max(mel_lens),pitches,energies,durations,attn_priors,spker_embeds,emotions,history_info)
        #上面代码只获得(texts,mels,pitches,energies,durations) 注意pitches,energies是来自于CompTransTTS的return中的p_targets和e_targets（音素级pitch和energy ）
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks, #(B,T)
            mel_masks,   #(B,D)
            src_lens,  #(B,) 存放音频的实际音素序列长度
            mel_lens,  #(B,) 存放音频的实际帧长
            attn_outs,
        ) = predictions
        """ output,  #(B,D,80)
            postnet_output, #(B,D,80)
            p_predictions, #prediction的形状为(Batchsize,T)，其中元素为模型预测的音素级的pitch
            e_predictions, #同理跟p_prediction一样
            log_d_predictions, #log_duration_prediction形状为(B,T),存放音素的（帧数+1）的log值
            d_rounded, #duration_rounded形状为(B,T),存放音素的帧数
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_outs, #attn_out = (attn_soft, attn_hard, attn_hard_dur, attn_logprob) ，含有四个元素的元组"""
        # ~的意思是按位取反，这样会将原本“padding 位置为 True” 的 mask 取反，变成 “有效（non‑padding）位置为 True” 的新 mask
        self.src_masks = src_masks = ~src_masks
        mel_masks = ~mel_masks


        if self.learn_alignment:
            attn_soft, attn_hard, attn_hard_dur, attn_logprob = attn_outs
            """attn_soft是一个B x 1 x D x T 的 attention 矩阵， 这个注意力矩阵是个概率，在(B,1,D,T)上对最后一维用softmax，(对于确定的b和d)attn[b,0,d,i] 就可以理解为“第 d 帧对应（或“对齐”到）第 i 个音素的概率
               attn_hard的形状 (B,1,D,T)，其中存储了batch中每个音频的帧音素的对齐路径信息(用0，1来表示) 并且包含padding的全0部分
               attn_hard_dur的形状 (B, T)，表示每个音素被模型预测的帧数(这个是根据注意力概率得到的模型的伪时长标签，用来当作真实时长标签)。
               attn_logprob是attn_soft取对数"""
            duration_targets = attn_hard_dur


        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        self.mel_masks = mel_masks = mel_masks[:, :mel_masks.shape[1]]

        #关闭对常量的梯度跟踪
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        #主要是masked_select函数的操作，取出有效数据构成一维数组
        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)  #.masked_select函数根据掩码，把 pitch_predictions 中所有有效（非 padding）的位置抽取出来(按先行后列的顺序拼成一个一维向量)，变成一个一维张量
            pitch_targets = pitch_targets.masked_select(src_masks) #同理
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        #预先初始化损失都为0，torch.zeros(1) 会创建一个只含有一个元素的一维张量，这个元素的值是0，默认数据类型是 torch.float32
        pitch_loss = energy_loss = torch.zeros(1).to(mel_targets.device)
        duration_loss = {
            "pdur": torch.zeros(1).to(mel_targets.device),
            "wdur": torch.zeros(1).to(mel_targets.device),
            "sdur": torch.zeros(1).to(mel_targets.device),
        }

        # 主要是masked_select函数的操作，取出有效数据构成一维数组
        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        #计算 Mel 谱损失，这两行都会返回一个标量张量
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        ctc_loss = bin_loss = torch.zeros(1).to(mel_targets.device)
        if self.learn_alignment:
            ctc_loss = self.sum_loss(attn_logprob=attn_logprob, in_lens=src_lens, out_lens=mel_lens)
            if step < self.binarization_loss_enable_steps:
                bin_loss_weight = 0.
                decoupling_weight = 0.0
            else:
                bin_loss_weight = min((step-self.binarization_loss_enable_steps) / self.binarization_loss_warmup_steps, 1.0) * 1.0
                decoupling_weight = min((step - self.binarization_loss_enable_steps) / self.binarization_loss_warmup_steps, 1.0) * 1.0
            bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight
            decoupling_loss=decoupling_loss * decoupling_weight

        total_loss = mel_loss + postnet_mel_loss + ctc_loss + bin_loss +decoupling_loss
        #当步数开始10000步，也就是开始验证的时候
        if step >= self.var_start_steps:
            pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
            energy_loss = self.mse_loss(energy_predictions, energy_targets)
            duration_loss = self.get_duration_loss(log_duration_predictions, duration_targets, texts)
            total_loss += sum(duration_loss.values()) + pitch_loss + energy_loss

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            ctc_loss,
            bin_loss,
            decoupling_loss
        )

    def get_duration_loss(self, dur_pred, dur_gt, txt_tokens):
        """
        :param dur_pred: [B, T], float, log scale
        :param txt_tokens: [B, T]
        :return:
        """
        losses = {}
        B, T = txt_tokens.shape
        nonpadding = self.src_masks.float()
        dur_gt = dur_gt.float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p_id in self.sil_ph_ids:
            is_sil = is_sil | (txt_tokens == p_id)
        is_sil = is_sil.float()  # [B, T_txt]

        # phone duration loss
        if self.loss_config["dur_loss"] == "mse":
            losses["pdur"] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction="none")
            losses["pdur"] = (losses["pdur"] * nonpadding).sum() / nonpadding.sum()
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        elif self.loss_config["dur_loss"] == "mog":
            return NotImplementedError
        elif self.loss_config["dur_loss"] == "crf":
            # losses["pdur"] = -self.model.dur_predictor.crf(
            #     dur_pred, dur_gt.long().clamp(min=0, max=31), mask=nonpadding > 0, reduction="mean")
            return NotImplementedError
        losses["pdur"] = losses["pdur"] * self.loss_config["lambda_ph_dur"]

        # use linear scale for sent and word duration
        if self.loss_config["lambda_word_dur"] > 0:
            word_id = (is_sil.cumsum(-1) * (1 - is_sil)).long()
            word_dur_p = dur_pred.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_pred)[:, 1:]
            word_dur_g = dur_gt.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_gt)[:, 1:]
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction="none")
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / (word_nonpadding.sum() + 1e-6)
            losses["wdur"] = wdur_loss * self.loss_config["lambda_word_dur"]
        if self.loss_config["lambda_sent_dur"] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction="mean")
            losses["sdur"] = sdur_loss.mean() * self.loss_config["lambda_sent_dur"]
        return losses


class ForwardSumLoss(nn.Module):
    """输入参数：attn_logprob, in_lens, out_lens，attn_logprob为（B,1,D,T）；in_lens为（B,）存放音频的音素序列长度；out_lens为（B，）存放音频实际帧长\n
     输出为"""
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)   #先做softmax再取对数
        """ nn.LogSoftmax(dim=3)"""
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        """nn.CTCLoss(zero_infinity=True)"""
        self.blank_logprob = blank_logprob
        """-1"""

    #sum_loss(attn_logprob=attn_logprob, in_lens=src_lens, out_lens=mel_lens)
    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob) #在最后一维（长度为 T 的那一维）左边补了 1 列，变成 (B, 1, D, T+1)，用-1填充
        """attn_logprob_padded的形状变为(B,1,D,T+1)"""

        total_loss = 0.0
        #对一个batch中的每个音频分别操作
        for bid in range(attn_logprob.shape[0]):
            #torch.arange(1, D+1) 生成一个从 1 到 D 的整型序列（不包含 0，因为 0 保留给 CTC 中的 blank）。
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0) #[1,2,3,...实际音素序列长度]unsqueeze(0)，target_seq的shape为[1,实际音素序列长度]
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]  #curr_logprob的形状为(实际帧长，1，实际音素序列长度+1)，这里的+1是CTC 中的 blank

            #curr_logprob[None]其实就是在张量的最前面插入一个新的维度变成了(1，实际帧长，1，实际音素序列长度+1)
            curr_logprob = self.log_softmax(curr_logprob[None])[0] #此时 curr_logprob变为了(实际帧长，1，实际音素序列长度+1)
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss


class BinLoss(nn.Module):
    """像是对动态规划的计算最优路径的损失,BinLoss 模块是用来度量“硬注意力”（binarized attention,二值化注意力矩阵，只有 0/1）和“软注意力”（模型输出的原始注意力分布,也就是概率矩阵）之间的一致性"""
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        return -log_sum / hard_attention.sum()
