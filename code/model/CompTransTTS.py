import os
import json
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random


from .modules import PostNet, VarianceAdaptor, ConversationalContextEncoder,TextPhonemeLevelModule,AcousticCoarseModule,AcousticFineGrainedModule,SpeakerEmbeddingLayer
from .modules import SubnetTCNSE,SequenceGLFK,FCFeatureExtractor,HistoryGuidedContrastiveLoss
from .modules import ReconstructionModule_text,SpecificReconstructionModule,OrthogonalityLossModule,calculate_decoupling_loss
from .modules import LanguageFocusedAttractor,SharedFeatureProcessor,GatedFusion
from utils.tools import get_mask_from_lengths



class CompTransTTS(nn.Module):
    """ CompTransTTS """

    def __init__(self, preprocess_config, model_config, train_config):
        super(CompTransTTS, self).__init__()
        self.model_config = model_config

        if model_config["block_type"] == "transformer":
            from .transformers.transformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "lstransformer":
        #     from .transformers.lstransformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "fastformer":
        #     from .transformers.fastformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "conformer":
        #     from .transformers.conformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "reformer":
        #     from .transformers.reformer import TextEncoder, Decoder
        else:
            raise ValueError("Unsupported Block Type: {}".format(model_config["block_type"]))

        self.encoder = TextEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, train_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],  #256
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"], #80
        )
        self.postnet = PostNet()

        self.speaker_emb = self.emotion_emb = None
        if model_config["multi_speaker"]:  #true
            self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]  #none
            if self.embedder_type == "none":
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))  #n_speaker=2
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],  #256
                )
            else:
                self.speaker_emb = nn.Linear(
                    model_config["external_speaker_dim"],
                    model_config["transformer"]["encoder_hidden"],
                )
        #同上面的得到说话人嵌入过程一模一样
        if model_config["multi_emotion"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "emotions.json"
                ),
                "r",
            ) as f:
                n_emotion = len(json.load(f))  #7
            self.emotion_emb = nn.Embedding(
                n_emotion,
                model_config["transformer"]["encoder_hidden"],  #256
            )
        self.history_type = model_config["history_encoder"]["type"]  #Guo

        if self.history_type != "none":
            if self.history_type == "Guo":
                self.context_encoder = ConversationalContextEncoder(preprocess_config, model_config)

                #TextPhonemeLevelModule,AcousticCoarseModule,AcousticFineGrainedModule,SpeakerEmbeddingLayer这四个自定义模块的初始化
                self.TextPhonemeLevel_encoder=TextPhonemeLevelModule()
                self.speakerembeddinglayer=SpeakerEmbeddingLayer()
                self.AcousticCoarse_encoder=AcousticCoarseModule()
                self.AcousticFineGrained_encoder=AcousticFineGrainedModule()
        self.subnet_text=SubnetTCNSE()
        self.subnet_audio=SubnetTCNSE()
        self.share_HGLFK=SequenceGLFK()
        self.criterion=HistoryGuidedContrastiveLoss()
        self.fusion=GatedFusion()
        self.share_fc_text=FCFeatureExtractor()
        self.share_fc_audio=FCFeatureExtractor()
        self.reconstruct_text=ReconstructionModule_text()
        self.reconstruct_audio=ReconstructionModule_text()
        self.spe_reconstruct_text=SpecificReconstructionModule(self.subnet_text)
        self.spe_reconstruct_audio=SpecificReconstructionModule(self.subnet_audio)
        self.OrthogonalityLoss=OrthogonalityLossModule()
        self.LFA=LanguageFocusedAttractor()
        self.sharedfeatureprocessor=SharedFeatureProcessor()
    def forward(
        self,
        speakers,  #(B,)，对应的说话人 ID
        texts,     #(B,T) 音素ID 序列,T为batch中的最大音素序列长度
        src_lens, #(B,) 存放音频的实际音素序列长度
        max_src_len,  #int或None？
        mels=None,  #(B,D,80)
        mel_lens=None, #(B,) 存放音频的实际帧长
        max_mel_len=None, #int 或 None
        p_targets=None, #(B, D)， 每帧的pitch 数值。D为batch中最大帧数
        e_targets=None,  #(B, D)， 每帧的energy 数值。D为batch中最大帧数
        d_targets=None,
        attn_priors=None, #(B,D,T)
        spker_embeds=None,  #(B,256)
        emotions=None,   #(B,) 情感 ID
        history_info=None,  #是一个元组，(text_emb, history_len, history_text_emb, history_speaker, history_phone_seq,history_phone_seq_mask,history_wav,history_wav_mask)#shape分别为(B,512)，(B,)，(B, 10, 512)，(B, 10)，，代码没改之前是这个，现在添了点东西
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None,
        id=1,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len) #src_masks为(B,T)的布尔型的mask
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )  #mel_masks为(B,D)的布尔型的mask

        #Transfprmer.py下的encoder
        texts, text_embeds = self.encoder(texts, src_masks) #都是(B,T,256)前面是经过transformer上下文的嵌入向量，后面那个只是根据id得到的嵌入向量

        #历史说话人编码
        his_speaker_emd=self.speakerembeddinglayer(history_info[3])  #(B,10,256)

        # Context Encoding
        #音素级的(细粒度)文本上下文编码
        TextPhonemeLevel_encodings=self.TextPhonemeLevel_encoder(history_info[4],history_info[5],texts,his_speaker_emd,history_info[4].shape[-1],id)   #(B,T,256)

        # 粗粒度音频上下文编码
        AcousticCoarse_encodings=self.AcousticCoarse_encoder(history_info[6],history_info[7],id)    #(B,256)

        # 细粒度音频上下文编码
        max_samples = history_info[6].shape[-1]
        total_stride = 320  # wav2vec2-base 的总下采样倍率
        max_hist_frames = math.ceil(max_samples / total_stride)
        AcousticFineGrained_encodings=self.AcousticFineGrained_encoder(history_info[6],history_info[7],texts,his_speaker_emd,max_hist_frames,id)   #(B,T,256)
        ##粗粒度文本上下文编码
        context_encodings = None
        if self.history_type != "none":
            if self.history_type == "Guo":
                (
                    text_embs,
                    history_lens,
                    history_text_embs,
                    history_speakers,
                ) = history_info[:4]

                #粗粒度文本上下文编码
                #我在这里去掉了
                """context_encodings为(B,256)"""

        Init_audio_feature=AcousticCoarse_encodings.unsqueeze(1).expand(-1, texts.shape[1], -1)+AcousticFineGrained_encodings
        Init_text_feature=TextPhonemeLevel_encodings  #+context_encodings.unsqueeze(1).expand(-1, texts.shape[1], -1)
        # 阶段一: 特征解耦 (Disentanglement)
        # ==============================================================================


        text_feature = self.share_fc_text(Init_text_feature)
        audio_feature = self.share_fc_audio(Init_audio_feature)
        # 1. 提取模态特定特征 (Sp)
        specific_feature_text = self.subnet_text(text_feature)  # Sp_L
        specific_feature_audio = self.subnet_audio(audio_feature)  # Sp_A
        # 2. 提取模态共享特征 (Sh)
        #    假设 share_HGLFK 接收两个模态作为输入，并输出一个融合后的共享特征。
        #    如果它输出两个特征，通常的做法是取平均: (common_text + common_audio) / 2
        common_feature = self.share_HGLFK(text_feature, audio_feature)  # Sh
        # ==============================================================================
        # 阶段二: 计算解耦损失 (Ld = Lr + Ls + Lo + Lm)
        # ==============================================================================

        # 1. 重构损失 (Lr)
        #    从分解后的特征重构出原始（或初步编码后）的特征
        #reconstructed_text, loss_r_text = self.reconstruct_text(common_feature, specific_feature_text,text_feature)
        #reconstructed_audio, loss_r_audio = self.reconstruct_audio(common_feature, specific_feature_audio,audio_feature)
        #loss_r = loss_r_text + loss_r_audio

        # 2. 特定重构损失 (Ls)
        #    验证重构特征中是否保留了足够的特定信息
        #_, loss_s_text = self.spe_reconstruct_text(reconstructed_text, specific_feature_text)
        #_, loss_s_audio = self.spe_reconstruct_audio(reconstructed_audio, specific_feature_audio)
        #loss_s = loss_s_text + loss_s_audio

        # 3. 正交性损失 (Lo)
        #    促使共享特征和特定特征相互独立
        #loss_o = self.OrthogonalityLoss(common_feature, specific_feature_text, specific_feature_audio)
        # ==============================================================================
        # 阶段三: 增强特定特征和共享特征 (Enhancement)
        # ==============================================================================
        HSp_t,HSp_a=self.LFA(specific_feature_text, specific_feature_audio)
        # 1. 增强共享特征，一主一辅 -> HSh
        #
        HSh = self.sharedfeatureprocessor(common_feature)
        loss_r=loss_s=loss_m=loss_o = torch.tensor(0.0, device=HSh.device)
        if id!=0:  #if id!=0: /False
            #4.计算损失lm
            file_path = os.path.join("./caption", f"batch_{id}.npy")
            data_numpy = np.load(file_path)
            history_tensor = torch.from_numpy(data_numpy).float().to(HSh.device)
            loss_m=self.criterion(HSh,history_tensor)
        decoupling_loss = calculate_decoupling_loss(loss_r, loss_s, loss_m, loss_o)
        # ==============================================================================
        # 阶段四: 融合

        # 2. 最终融合
        #    使用门控融合模块融合所有增强后的特征
        final_fused_feature = self.fusion(HSh, HSp_t, HSp_a)



        speaker_embeds = None
        if self.speaker_emb is not None:
            if self.embedder_type == "none":
                speaker_embeds = self.speaker_emb(speakers) # [B, 256]
            else:
                assert spker_embeds is not None, "Speaker embedding should not be None"
                speaker_embeds = self.speaker_emb(spker_embeds) # [B, H]

        emotion_embeds = None  #同上面的speaker_embeds一样
        if self.emotion_emb is not None:
            emotion_embeds = self.emotion_emb(emotions)  #[B, 256]

        (
            output,  #(B,D,256)
            p_targets, #pitch_target的shape为(B,T) T为batch中音频的最大音素序列长度，每个元素为对应音素的音素级pitch就是(音素对应的) 插值过的帧级pitch取平均
            p_predictions, #prediction的形状为(Batchsize,T)，其中元素为log(预测的音素的帧的时长+1)
            e_targets, #同理跟p_targets一样，只不过是energy
            e_predictions, #同理跟p_predictions一样
            log_d_predictions, #log_duration_prediction形状为(B,T),存放音素的（帧数+1）的log值
            d_rounded,  #duration_rounded形状为(B,T),存放音素的帧数
            mel_lens, #mel_len为一个long类型的张量，张量形状为(batchsize,) 是一个一维张量，存放的是一个batch中的音频帧长
            mel_masks, #一个形状为 (B, D) 的布尔掩码张量 mask，D为一个batch中的最大帧长，用来屏蔽掉那些“填充”出来的无效位置
            attn_outs, #attn_out = (attn_soft, attn_hard, attn_hard_dur, attn_logprob) ，含有四个元素的元组
        ) = self.variance_adaptor(
            speaker_embeds,
            emotion_embeds,
            final_fused_feature,
            texts,
            text_embeds,
            src_lens,
            src_masks,
            mels,
            mel_lens,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            attn_priors,
            p_control,
            e_control,
            d_control,
            step,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)  #(B,D,256)->(B,D,80),初始梅尔频谱经过线性变换

        postnet_output = self.postnet(output) + output    #经过卷积加和
        """postnet_output为(B,D,80)"""

        return (
            output,  #(B,D,80)
            postnet_output, #(B,D,80)
            p_predictions, #prediction的形状为(Batchsize,T)，其中元素为模型预测的音素级的pitch
            e_predictions, #同理跟p_prediction一样
            log_d_predictions, #log_duration_prediction形状为(B,T),存放音素的（帧数+1）的log值
            d_rounded, #duration_rounded形状为(B,T),存放音素的帧数
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_outs, #attn_out = (attn_soft, attn_hard, attn_hard_dur, attn_logprob) ，含有四个元素的元组
            p_targets, #pitch_target的shape为(B,T) T为batch中音频的最大音素序列长度，每个元素为对应音素的音素级pitch就是(音素对应的) 插值过的帧级pitch取平均
            e_targets,  #同理跟p_targets一样，只不过是energy
            decoupling_loss  #复合损失
        )
