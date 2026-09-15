import os
import random
import re
import json
import copy

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.stats import betabinom
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio
from text import text_to_sequence, sequence_to_text, grapheme_to_phoneme
from utils.tools import get_phoneme_level_pitch, get_phoneme_level_energy
from g2p_en import G2p
from sentence_transformers import SentenceTransformer


class Preprocessor:
    def __init__(self, preprocess_config, model_config, train_config):
        #初始化 Python 内置 random 模块的伪随机数生成器，并且把它的“种子（seed）”设成配置文件里 train_config['seed'] 指定的那个固定数值。我们往往希望“随机”操作（比如 random.sample 划分验证集、打乱列表等）每次都能得到完全相同的结果，
        random.seed(train_config['seed'])
        self.config = preprocess_config
        self.dataset = preprocess_config["dataset"]
        #初始化两个空集合，用来在预处理过程中动态收集说话人 ID 和情感标签。
        self.speakers = set()
        self.emotions = set()
        self.sub_dir = preprocess_config["path"]["sub_dir_name"]
        """sub_dir=data
        """
        self.data_dir = preprocess_config["path"]["corpus_path"]
        """
        data_dir=D:/PythonProject/DailyTalk-main/ """
        self.in_dir = os.path.join(preprocess_config["path"]["raw_path"], self.sub_dir)
        """in_dir=./raw_data/DailyTalk/data"""
        self.out_dir = preprocess_config["path"]["preprocessed_path"]
        """结果输出目录out_dir=./preprocessed_data/DailyTalk"""
        self.val_size = 2
        """随机划分时用于验证集的对话数量（这里硬编码为 2 条）"""
        # self.val_size = preprocess_config["preprocessing"]["val_size"]  ，这里是32
        self.val_dialog_ids = self.get_val_dialog_ids()
        """调用get_val_dialog_ids() 随机抽取2条对话 ID，保存到 self.val_dialog_ids。"""
        self.metadata = self.load_metadata()
        """metadata=将metadata.json文件读入python对象中，一般为字典"""
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        """sampling_rate=22050"""
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        """hop_length: 256   帧移，每隔 hop_length 个采样点滑动一次，产生一帧新的特征向量。"""
        self.filter_length = preprocess_config["preprocessing"]["stft"]["filter_length"]
        """filter_length: 1024  # 短时能量计算的帧长度（帧长度也就是采样点个数，每一帧包括1024个采样点），需与 mel.n_mel_channels 匹配（如 80 通道对应 1024）"""
        self.trim_top_db = preprocess_config["preprocessing"]["audio"]["trim_top_db"]
        """trim_top_db: 23  静音裁剪阈值（dB），信号低于 23dB 视为静音。"""
        self.beta_binomial_scaling_factor = preprocess_config["preprocessing"]["duration"]["beta_binomial_scaling_factor"]
        """beta_binomial_scaling_factor: 1.0   # 调整持续时间分布的平滑度，控制持续时间分布的形状（值越大，分布越集中）。  常见取值：1.0（默认）、0.5（更分散）、2.0（更集中）。"""
        self.g2p = G2p()
        """g2p是音素转换工具，Grapheme-to-Phoneme，将字母转为音素 \n
            输入是Python 字符串（str）可以是一个或多个英文单词。\n
            输出是：Python 列表（list），列表中的每个元素都是一个 音素标记（字符串）。['HH', 'AH0', 'L', 'OW1', 'W', 'ER1', 'L', 'D'] 前四个为hello，后四个为world"""
        self.text_embbeder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        """text_embbeder为多语言版本的句向量编码器。\n
            利用 HuggingFace 提供的 SentenceTransformer，实例化一个多语言版本的句向量编码器。\n
            后面在处理每条对话文本时，会调用它的 encode(...) 方法，把原始一句话映射成一个定长的浮点向量，作为文本特征输入。\n
            encode 一般要求参数是 List[str]，即使你只想编码一条，也要把它放到列表里。调用 encode([raw_text]) 会返回一个列表或二维数组，形状通常是 (batch_size, 512)\n
        """

        #校验配置合法性,确保配置文件中对 pitch（基频）和 energy（能量）特征提取所使用的粒度是 “音素级”（phoneme_level）或者 “帧级”（frame_level） 之一，否则程序直接抛错终止。
        assert preprocess_config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert preprocess_config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        """pitch_phoneme_averaging是为true的，表示后面要把同一音素对应的多帧基频(音高，单位为hz)做平均（音素级特征）；"""
        self.energy_phoneme_averaging = (
            preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )
        """energy_phoneme_averaging也是为true的，表示后面要把同一音素对应的多帧能量做平均（音素级特征）"""

        self.pitch_normalization = preprocess_config["preprocessing"]["pitch"]["normalization"]
        """pitch_normalization为true,对已提取出来的基频(音高，单位为hz)做 (x - mean)/std 归一化。"""
        self.energy_normalization = preprocess_config["preprocessing"]["energy"]["normalization"]
        """energy_normalization为true,对已提取出来的能量做 (x - mean)/std 归一化。"""
        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"], #1024
            preprocess_config["preprocessing"]["stft"]["hop_length"],   #256
            preprocess_config["preprocessing"]["stft"]["win_length"],    #1024
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],   #80
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],   #22050
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],   #0
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],    #8000
        )
        """STFT是实例化的一个TacotronSTFT对象，作用是将把原始的音频波形转换成 Mel 频谱（以及顺带的能量等特征）\n
        例如：mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)，返回形状为 (n_mel_channels=80, T_frames=xx) 的 Mel 频谱矩阵，以及长度为 T_frames 的能量向量。"""
        self.val_dialog_ids_prior_frame = self.get_val_dialog_ids_prior(os.path.join(self.out_dir, "val_frame.txt"))
        """如果在上一次运行或用户预先生成过 val_frame.txt（帧级验证集 ID 列表）或 val_phone.txt（音素级验证集 ID 列表），get_val_dialog_ids_prior(...) 会读取这些文件并返回一个 ID 列表赋值给val_dialog_ids_prior_phone
                或val_dialog_ids_prior_frame；否则返回 None，后面才改用随机划分的逻辑"""
        self.val_dialog_ids_prior_phone = self.get_val_dialog_ids_prior(os.path.join(self.out_dir, "val_phone.txt"))
        """如果在上一次运行或用户预先生成过 val_frame.txt（帧级验证集 ID 列表）或 val_phone.txt（音素级验证集 ID 列表），get_val_dialog_ids_prior(...) 会读取这些文件并返回一个 ID 列表赋值给val_dialog_ids_prior_phone
        或val_dialog_ids_prior_frame；否则返回 None，后面才改用随机划分的逻辑"""

    def get_val_dialog_ids_prior(self, val_prior_path):
        """如果preprocessed_data/DailyTalk/val_phone.txt存在，那么返回的就是作者预设的验证集对话id，是一个列表，否则返回null"""
        val_dialog_ids_prior = set()  #创建一个空的 set，用于存储从文件中解析出的对话 ID，利用集合可以去重。
        if os.path.isfile(val_prior_path):
            #文件存在
            print("Load pre-defined validation set...")
            with open(val_prior_path, "r", encoding="utf-8") as f:
                #逐行读取文件内容，每行 m 对应 val_frame.txt 或 val_phone.txt 中的一行
                for m in f.readlines():
                    val_dialog_ids_prior.add(int(m.split("|")[0].split("_")[-1].strip("d")))
            return list(val_dialog_ids_prior)
        else:
            return None

    def get_val_dialog_ids(self):
        """调用val_dialog_ids = random.sample(range(data_size), k=self.val_size)，data_size是对话数量
        range(data_size)：生成从 0 到 data_size-1 的整数序列，对应每个对话的索引。
           random.sample(..., k=self.val_size)：从这个索引序列中随机且不重复地抽取 k 个元素，k 的值由实例属性 self.val_size 决定（一般在初始化里设为 2，表示抽 2 条对话做验证）。
           抽出的那几个整数列表，就代表了哪些对话被划为验证集，保存在变量 val_dialog_ids 中。 """
        data_size = len(os.listdir(self.in_dir))    #出预处理器内部 self.in_dir 目录下的所有子目录或文件名。,data_size代表总共多少个对话
        val_dialog_ids = random.sample(range(data_size), k=self.val_size)
        # print("val_dialog_ids:", val_dialog_ids)
        return val_dialog_ids

    def load_metadata(self):
        """读取metadata.json文件，解析成对应的 Python 对象（通常是嵌套的 dict 和 list）。"""
        with open(os.path.join(self.data_dir, "metadata.json"),encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata

    def build_from_path(self):
        """1.在preprocessed_data/DailyTalk下创建共 9 个子目录：mel_frame、mel_phone、pitch_frame、pitch_phone、energy_frame、energy_phone、duration、attn_prior"""
        os.makedirs((os.path.join(self.out_dir, "text_emb")), exist_ok=True)  #exist_ok=True保证已存在时不报错
        os.makedirs((os.path.join(self.out_dir, "mel_frame")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mel_phone")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch_frame")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch_phone")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_frame")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_phone")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "attn_prior")), exist_ok=True)

        print("Processing Data ...")
        filtered_out_dialog_frame = set()
        """filtered_out_dialog_frame是用于记录“被过滤掉”的对话ID——针对帧级。如果某个对话的所有样本都没有有效特征（比如全静音或对齐失败），就会把该对话 ID 加入对应的集合。"""
        filtered_out_dialog_phone = set()
        """filtered_out_dialog_frame是用于记录“被过滤掉”的对话ID——针对音素级。如果某个对话的所有样本都没有有效特征（比如全静音或对齐失败），就会把该对话 ID 加入对应的集合。"""
        train_frame = list()
        """存放训练集的帧级（frame-level）的样本列表，每个元素是字符串 "basename|speaker|text_frame|raw_text|emotion"。"""
        val_frame = list()
        """存放验证集的帧级（frame-level）的样本列表，每个元素是字符串 "basename|speaker|text_frame|raw_text|emotion"。"""
        train_phone = list()
        """存放训练集的音素级（phone-level）的样本列表，每个元素是字符串 "basename|speaker|text_phone|raw_text|emotion"。"""
        val_phone = list()
        """存放验证集的音素级（phone-level）的样本列表，每个元素是字符串 "basename|speaker|text_phone|raw_text|emotion"。"""
        n_frames = 0
        """用于累积所有有效语音(要么帧级满足，要么音素级满足，或者都满足)的帧总数，方便后面计算整个数据集的总时长(总帧数)。"""
        max_seq_len = -float('inf')
        """记录迄今为止遇到的有效语音的最大帧数，也可以理解为最长序列长度（以帧为单位）。初始设置为负无穷，一旦处理到任何一个样本，就会更新为该样本的实际帧数。"""
        pitch_frame_scaler = StandardScaler()
        pitch_phone_scaler = StandardScaler()
        energy_frame_scaler = StandardScaler()
        energy_phone_scaler = StandardScaler()
        """分别是四个 sklearn.preprocessing.StandardScaler 实例，用来在线（partial）累积和计算：帧级 pitch 的均值与方差,音素级 pitch 的均值与方差,帧级 energy 的均值与方差,音素级 energy 的均值与方差(均值的计算，是迭代的一个音频的，全部帧的pitch的均值)\n
           后续每处理完一个样本，就用 scaler.partial_fit() 累加该样本的数值，最后可以直接从 scaler.mean_ 和 scaler.scale_ 里读出全数据集的均值和标准差，用于归一化或统计报告。
        """

        def partial_fit(scaler, value):
            """调用增量训练接口，逐批累加所有样本的数据统计（内部更新 .mean_、.var_、.scale_ 等属性）。但是将value一个一维非空数组
                转换为改成 N×1 的二维数组。因为StandardScaler 要求输入形状为 (样本数, 特征数)。"""
            if len(value) > 0:
                scaler.partial_fit(value.reshape((-1, 1)))

        def compute_stats(pitch_scaler, energy_scaler, pitch_dir="pitch", energy_dir="energy"):
            """pitch_dir="pitch", energy_dir="energy",两个参数的默认值是字符串中的内容 \n
               实际代码执行对preprocessed_data/DailyTalk/pitch_frame处理，
               pitch_dir="pitch_phone/pitch_frame",
               energy_dir="energy_phone/energy_phone_frame"\n
               最后结果返回(归一化后的pitch_min, 归一化后的pitch_max, pitch_mean, pitch_std), (归一化后的energy_min, 归一化后的energy_max, energy_mean, energy_std)其中min和max都是所有处理音频归一化后的最大值和最小值，均值也是所有样本的
            """
            if self.pitch_normalization:
                pitch_mean = pitch_scaler.mean_[0]
                """pitch_scaler.mean_返回的是一个数组，只有一个元素，就是所有样本的pitch特征均值"""
                pitch_std = pitch_scaler.scale_[0]
                """所有样本的pitch特征标准差"""
            else:
                # A numerical trick to avoid normalization...
                pitch_mean = 0
                pitch_std = 1
            if self.energy_normalization:
                energy_mean = energy_scaler.mean_[0]
                energy_std = energy_scaler.scale_[0]
            else:
                energy_mean = 0
                energy_std = 1

            pitch_min, pitch_max = self.normalize(
                os.path.join(self.out_dir, pitch_dir), pitch_mean, pitch_std
            )
            energy_min, energy_max = self.normalize(
                os.path.join(self.out_dir, energy_dir), energy_mean, energy_std
            )
            return (pitch_min, pitch_max, pitch_mean, pitch_std), (energy_min, energy_max, energy_mean, energy_std)

        # Compute pitch, energy, duration, and mel-spectrogram
        # speakers = self.speakers.copy()
        """for循环做了什么：1.生成preprocessed_data/DailyTalk下的这些目录下的npy文件
                         2. 归一化
                        3 划分训练集和验证集，用对话id划分"""
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))): # here, speaker is actually a dialog_id
            # if len(self.speakers) == 0:
            #     speakers[speaker] = i
            """tqdm(...)把这个列表包装成一个可迭代对象，并在终端显示进度条。每循环一次，进度条就会更新。
            在遍历的同时为每个元素生成一个从 0 开始的索引，迭代时会产出 (索引, 元素值) 这样的二元组。
            """
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue
                #对raw_data/DailyTalk/data/0下的.wav文件处理（举例是0号对话，最终是对所有对话）
                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                """tg_path就是和当前处理音频对应的preprocessed_data/DailyTalk/TextGrid/${对话id}/${basename}.TextGrid """
                (
                    info_frame,
                    info_phone,
                    pitch_frame,
                    pitch_phone,
                    energy_frame,
                    energy_phone,
                    n,
                ) = self.process_utterance(tg_path, speaker, basename)
                if info_frame is None and info_phone is None:  #当前音频样本音素级和帧级都不合格，可能是静音太多
                    filtered_out_dialog_frame.add(int(speaker))
                    filtered_out_dialog_phone.add(int(speaker))
                    continue
                else:
                    # Save frame level information
                    #  根据 speaker，也就是对话id值，决定放到 train_frame 还是 val_frame(不在预设的val_frame的就放到train_frame中)
                    if info_frame is not None:
                        if self.val_dialog_ids_prior_frame is not None:
                            if int(speaker) not in self.val_dialog_ids_prior_frame:
                                train_frame.append(info_frame)
                            else:
                                val_frame.append(info_frame)
                        else:
                            if int(speaker) not in self.val_dialog_ids:
                                train_frame.append(info_frame)
                            else:
                                val_frame.append(info_frame)
                        #用这条样本的 pitch_frame / energy_frame 调用 partial_fit，累积均方统计
                        partial_fit(pitch_frame_scaler, pitch_frame)
                        partial_fit(energy_frame_scaler, energy_frame)
                    else:
                        filtered_out_dialog_frame.add(int(speaker)) #当前音频样本帧级不合格，可能是静音太多
                    # Save phone level information
                    #类似于帧级处理，对音素级做相同的处理，根据 speaker，也就是对话id值，决定放到 train_phonee 还是 val_phone(不在预设的val_phone 的就放到train_phone中)
                    if info_phone is not None:
                        if self.val_dialog_ids_prior_phone is not None:
                            if int(speaker) not in self.val_dialog_ids_prior_phone:
                                train_phone.append(info_phone)
                            else:
                                val_phone.append(info_phone)
                        else:
                            if int(speaker) not in self.val_dialog_ids:
                                train_phone.append(info_phone)
                            else:
                                val_phone.append(info_phone)

                        partial_fit(pitch_phone_scaler, pitch_phone)
                        partial_fit(energy_phone_scaler, energy_phone)
                    else:
                        filtered_out_dialog_phone.add(int(speaker))

                    if n > max_seq_len:
                        # 迄今为止遇到的有效语音的最大帧长
                        max_seq_len = n
                    #累积有效语音的帧长和
                    n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary， 进行标准化
        pitch_frame_stats, energy_frame_stats = compute_stats(
            pitch_frame_scaler,
            energy_frame_scaler,
            pitch_dir="pitch_frame",
            energy_dir="energy_frame",
        )
        pitch_phone_stats, energy_phone_stats = compute_stats(
            pitch_phone_scaler,
            energy_phone_scaler,
            pitch_dir="pitch_phone",
            energy_dir="energy_phone",
        )

        # Save files
        # with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
        #     f.write(json.dumps(speakers))

        #此时speakers={0，1}，生成speaker.json
        if len(self.speakers) != 0:
            speaker_dict = dict()
            for i, speaker in enumerate(list(self.speakers)):
                speaker_dict[speaker] = int(speaker)
            with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
                f.write(json.dumps(speaker_dict))
        #生成emotions.json
        if len(self.emotions) != 0:
            emotion_dict = dict()
            for i, emotion in enumerate(list(self.emotions)):
                emotion_dict[emotion] = i
            with open(os.path.join(self.out_dir, "emotions.json"), "w") as f:
                f.write(json.dumps(emotion_dict))

        #将数据集的pitch状态和energy状态写入stats.json，(归一化后的最小值，归一化后的最大值，均值，方差，最大帧长的顺序)
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch_frame": [float(var) for var in pitch_frame_stats],
                "pitch_phone": [float(var) for var in pitch_phone_stats],
                "energy_frame": [float(var) for var in energy_frame_stats],
                "energy_phone": [float(var) for var in energy_phone_stats],
                "max_seq_len": max_seq_len
            }
            f.write(json.dumps(stats))

        print(
            #打印的是n_frames 是所有音频样本的帧数总和；乘以每帧对应的采样点数 hop_length，再除以采样率 sampling_rate 得到总秒数，最后除以 3600 转成小时。这样可以直观地看到：这批数据总共相当于多少小时的音频，方便评估和报告。
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        #随机打乱顺序
        random.shuffle(train_frame)
        random.shuffle(train_phone)
        #保证非空
        train_frame = [r for r in train_frame if r is not None]
        train_phone = [r for r in train_phone if r is not None]
        #保证非空
        val_frame = [r for r in val_frame if r is not None]
        val_phone = [r for r in val_phone if r is not None]
        # Filter out incomplete dialog in all train & val set
        filtered_out_dialog_frame, filtered_out_dialog_phone = list(filtered_out_dialog_frame), list(filtered_out_dialog_phone)
        #保证不是不合格的音频样本
        train_frame = [r for r in train_frame if int(r.split("|")[0].split("_")[-1].strip("d")) not in filtered_out_dialog_frame]
        train_phone = [r for r in train_phone if int(r.split("|")[0].split("_")[-1].strip("d")) not in filtered_out_dialog_phone]
        val_frame = [r for r in val_frame if int(r.split("|")[0].split("_")[-1].strip("d")) not in filtered_out_dialog_frame]
        val_phone = [r for r in val_phone if int(r.split("|")[0].split("_")[-1].strip("d")) not in filtered_out_dialog_phone]
        # Sort validation set by dialog，将验证集排序，按照（对话id，当前音频在对话中的id）排序，即先按对话 ID（d12 中的 12），再按同一对话中语句在对话里的序号（0_1_d12 中的 0）排序，确保验证数据按对话顺序读取。
        val_frame = sorted(val_frame, key=lambda x: (int(x.split("|")[0].split("_")[-1].lstrip("d")), int(x.split("|")[0].split("_")[0])))
        val_phone = sorted(val_phone, key=lambda x: (int(x.split("|")[0].split("_")[-1].lstrip("d")), int(x.split("|")[0].split("_")[0])))

        # Write metadata
        #写入帧级过滤对话id
        with open(os.path.join(self.out_dir, "filtered_out_dialog_frame.txt"), "w", encoding="utf-8") as f:
            for m in sorted(filtered_out_dialog_frame):
                f.write(str(m) + "\n")
        # 写入音素级过滤对话id
        with open(os.path.join(self.out_dir, "filtered_out_dialog_phone.txt"), "w", encoding="utf-8") as f:
            for m in sorted(filtered_out_dialog_phone):
                f.write(str(m) + "\n")
        #生成train_frame.txt文件，内容是音频的信息
        with open(os.path.join(self.out_dir, "train_frame.txt"), "w", encoding="utf-8") as f:
            for m in train_frame:
                f.write(m + "\n")
        # 生成val_frame.txt文件，内容是音频的信息
        with open(os.path.join(self.out_dir, "val_frame.txt"), "w", encoding="utf-8") as f:
            for m in val_frame:
                f.write(m + "\n")
        # 生成train_phone.txt文件，内容是音频的信息
        with open(os.path.join(self.out_dir, "train_phone.txt"), "w", encoding="utf-8") as f:
            for m in train_phone:
                f.write(m + "\n")
        # 生成val_phone.txt文件，内容是音频的信息
        with open(os.path.join(self.out_dir, "val_phone.txt"), "w", encoding="utf-8") as f:
            for m in val_phone:
                f.write(m + "\n")

        return (train_frame ,train_phone, val_frame ,val_phone)

    def load_audio(self, wav_path):
        """有三个返回值wav_raw.astype(np.float32), wav.astype(np.float32), int(duration)，第一个为原始波形（一维 float32 数组），第二个为去静音后波形（一维 float32 数组），第三个为去静音后对应的帧数（整数）"""
        wav_raw, _ = librosa.load(wav_path, self.sampling_rate)
        """返回值wav_raw 是一个一维的浮点型 NumPy 数组（原始波形样本）,wav_raw 长度就是音频时长（秒）乘以 22050。"""
        _, index = librosa.effects.trim(wav_raw, top_db=self.trim_top_db, frame_length=self.filter_length, hop_length=self.hop_length)
        """index 是一个二元元组 (start_sample, end_sample)，表示将开头和结尾裁剪后保留的样本索引范围。start：第一个“非静音帧”对应的采样点索引，end：最后一个“非静音帧”结束后的采样点索引"""
        wav = wav_raw[index[0]:index[1]]
        duration = (index[1] - index[0]) / self.hop_length  #duration为这段音频有多少帧，要向下取整
        return wav_raw.astype(np.float32), wav.astype(np.float32), int(duration)

    #tg_path就是preprocessed_data/DailyTalk/TextGrid/${对话id}/${basename}.TextGrid
    def process_utterance(self, tg_path, speaker, basename):
        """传入参数：tg_path就是preprocessed_data/DailyTalk/TextGrid/${对话id}/${basename}.TextGrid ，speaker=对应语音的对话id，basename就是对应语音的basename\n
           返回一个7个元素的元组，第一个为Frame-level信息:basename|speaker|text_frame|raw_text|emotion，其中text_frame是直接调用包生成的音素序列,
                                  第二个为Phone-level信息：basename|speaker|text_phone|raw_text|emotion，其中text_phone是根据TextGrid文件生成的
                                  第三个为去掉离群值后的 Frame-level pitch，是一个一维数组，长度为duration也就是帧数
                                  第四个为去掉离群值后的 Phone-level pitch，是一个一维数组，长度为duration也就是帧数
                                  第五个为去掉离群值后的 Frame-level energy，是一个一维数组，长度为duration也就是帧数
                                  第六个为去掉离群值后的 Phone-level energy，是一个一维数组，长度为duration也就是帧数
                                  第七个元素为  在帧级梅尔谱的帧数和音素级梅尔谱的帧数之间取最大值，用来告诉调用者这一条音频最终有多少帧。"""
        phone_out_exist, frame_out_exist = True, True

        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        """./ raw_data/ DailyTalk/ data/对话id/.wav文件"""
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        """./ raw_data/ DailyTalk/ data/对话id/.lab文件"""
        #解析basename
        speaker = basename.split("_")[1]
        dialog_id = basename.split("_")[-1].lstrip("d")
        uttr_id = basename.split("_")[0]
        emotion = self.metadata[dialog_id][uttr_id]["emotion"]
        if emotion == "no emotion":
            emotion = "none"
        self.speakers.add(speaker)
        self.emotions.add(emotion)

        wav_raw, wav, duration = self.load_audio(wav_path)
        """ wav为去静音后波形（一维 float32 数组），duration为去静音后对应的帧数（整数） """

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")
        phone = grapheme_to_phoneme(raw_text, self.g2p)  #文本转为音素序列，phone是一个list[str]
        phones = "{" + "}{".join(phone) + "}"
        """phones是音素序列的字符串，类似于phones == "{HH}{AH0}{L}{OW1}",经过正则 \{[^\w\s]?\} 会匹配 {} 或者 {,}、{.}、{?} 这种内部只有一个非字母数字空白的情况，把它们统一替换成 {sp}，表示一个“silence/pause”帧"""
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)    #正则化匹配
        text_frame = phones.replace("}{", " ")
        """text_frame == "{HH AH0 L OW1}",将phones又变成空格分割"""

        # Text embedding
        text_emb = self.text_embbeder.encode([raw_text])[0]   #因为batch是1，返回的是一个二维数组，所以是[0]

        # Compute fundamental frequency
        # frame_period为帧间隔，即两次相邻 F0(pitch,基频) 估计之间的时间间隔，用ms为单位
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        """pitch：长度约为 n_samples/hop_length的一维数组(计算结果也就是duration，duration=帧数)，每个元素是对应帧的基频值（Hz），如果该帧无声或无法估计，就会是 0
            t：同样长度的时间轴数组（单位秒），标出每个 pitch[i] 对应的时刻。"""

        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)  #长度不变，只是对上面的数值进行精修，更准确
        pitch = pitch[: duration]  #一般情况没有必要，但是dio 在边界会有四舍五入或多估少估几帧的可能，这样保证和duration相等
        if np.sum(pitch != 0) <= 1:
            #如果“有声”帧数量不多于 1（即几乎全是静音或 F₀ 无效的帧），就认为这一条语音在帧级上“没有有效特征”，将 frame_out_exist 标记为 False，后续就不会对它做帧级特征保存、统计等处理。
            frame_out_exist = False
        else:
            # Compute mel-scale spectrogram and energy
            mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT) #  mel_spectrogram: 形状 (80, T) 的二维数组，T为STFT算出来的帧数，energy:形状 (T,) 的一维数组，每个元素对应每帧的声学能量
            mel_spectrogram = mel_spectrogram[:, : duration]    #mel_spectrogram切片匹配维度为(80, duration),实际上存入的是它的转置(duration,80)
            energy = energy[: duration]    #同理，维度为(duration,)

            # Compute alignment prior
            attn_prior = self.beta_binomial_prior_distribution(
                mel_spectrogram.shape[1],  # T：duration （mel 频谱的时间维度）
                len(phone),  # N：音素（phoneme）的数量
                self.beta_binomial_scaling_factor,  # α：平滑/拉伸因子
            )
            """为每条语音生成一个 注意力先验（alignment prior） 矩阵，用来指导模型在 “第 n 个音素” 对应 “第 t 帧” 时刻的粗略对齐概率。\n
            attn_prior 通常是一个形状为 (N, T) 的 NumPy 数组，行对应第 0 到第 N−1 个音素；列对应第 0 到第 T−1 帧，每个元素 attn_prior[n, t] 表示 “第 n 个音素” 在第 t 帧上的 先验注意力权重。训练时模型会在这个先验基础上，再学习更精细的对齐分布。"""
            # Frame-level variance(pitch、energy、mel-spectrogram这三个都是帧级特征)
            #深拷贝帧级特征，为了先把当前这版“帧级特征”（pitch/energy/梅尔谱）存一份备份，后续无论怎么处理原数组，都能保证备份是干净、完整的，用来保存成 .npy 文件。
            pitch_frame, energy_frame = copy.deepcopy(pitch), copy.deepcopy(energy)
            mel_spectrogram_frame = copy.deepcopy(mel_spectrogram)

            # Save files(text_emb,attn_prior,pitch_frame,mel_frame,energy_frame)
            """将特征和先验注意力矩阵保存在.npy文件中，text_emb,attn_prior,pitch_frame,mel_frame,energy_frame"""
            text_emb_filename = "{}-text_emb-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "text_emb", text_emb_filename), text_emb)

            attn_prior_filename = "{}-attn_prior-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "attn_prior", attn_prior_filename), attn_prior)

            pitch_frame_filename = "{}-pitch-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "pitch_frame", pitch_frame_filename), pitch_frame)

            energy_frame_filename = "{}-energy-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "energy_frame", energy_frame_filename), energy_frame)

            mel_frame_filename = "{}-mel-{}.npy".format(speaker, basename)
            np.save(
                os.path.join(self.out_dir, "mel_frame", mel_frame_filename),
                mel_spectrogram_frame.T,
            )


        # Supervised duration features(获取音素级的帧级特征(pitch,energy,梅尔频谱)，方法和上面一模一样)
        if os.path.exists(tg_path):
            # Get alignments
            textgrid = tgt.io.read_textgrid(tg_path)

            #textgrid.get_tier_by_name("phones")，返回一个 Tier 对象，这个 Tier 对象代表了名字为 "phones" 的那一层注释，你可以从它那里拿到所有的音素区间（或打点式标注）的起止时间和文本标签。
            phone, duration, start, end = self.get_alignment(
                textgrid.get_tier_by_name("phones")
            )
            text_phone = "{" + " ".join(phone) + "}"
            """text_phone="{音素1 音素2 ...}",将音素序列以空格分割,然后转为字符串"""
            if start >= end:
                #判断是否音频有效
                phone_out_exist = False
            else:
                # Read and trim wav files
                wav, _ = librosa.load(wav_path, self.sampling_rate)
                wav = wav.astype(np.float32)
                wav = wav[
                    int(self.sampling_rate * start) : int(self.sampling_rate * end)
                ]
                #按照得到的第一个非静音音素和最后一个非静音音素，重新裁剪wav，变成非静音的wav

                # Compute fundamental frequency
                pitch, t = pw.dio(
                    wav.astype(np.float64),
                    self.sampling_rate,
                    frame_period=self.hop_length / self.sampling_rate * 1000,
                )
                pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

                pitch = pitch[: sum(duration)]   #因为此时duration已经变成了int列表，元素为每个音素的帧长，而不是整个音频的帧长，因此需要用sum
                if np.sum(pitch != 0) <= 1:
                    phone_out_exist = False
                else:
                    # Compute mel-scale spectrogram and energy
                    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
                    mel_spectrogram = mel_spectrogram[:, : sum(duration)]
                    energy = energy[: sum(duration)]

                    # Phone-level variance
                    pitch_phone, energy_phone = get_phoneme_level_pitch(duration, pitch), get_phoneme_level_energy(duration, energy)
                    mel_spectrogram_phone = copy.deepcopy(mel_spectrogram)

                    # Save files
                    """将特征和先验注意力矩阵保存在.npy文件中，text_emb,duration,pitch_phone,mel_phone,energy_phone"""
                    if not frame_out_exist:
                        text_emb_filename = "{}-text_emb-{}.npy".format(speaker, basename)
                        np.save(os.path.join(self.out_dir, "text_emb", text_emb_filename), text_emb)

                    dur_filename = "{}-duration-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

                    pitch_phone_filename = "{}-pitch-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "pitch_phone", pitch_phone_filename), pitch_phone)

                    energy_phone_filename = "{}-energy-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "energy_phone", energy_phone_filename), energy_phone)

                    mel_phone_filename = "{}-mel-{}.npy".format(speaker, basename)
                    np.save(
                        os.path.join(self.out_dir, "mel_phone", mel_phone_filename),
                        mel_spectrogram_phone.T,
                    )
        else:
            phone_out_exist = False

        if not phone_out_exist and not frame_out_exist:
            return tuple([None]*7)
        else:
            """返回一个7个元素的元组，第一个为Frame-level 文本信息:basename|speaker|text_frame|raw_text|emotion，其中text_frame是直接调用包生成的音素序列,
                                  第二个为Phone-level 文本信息：basename|speaker|text_phone|raw_text|emotion，其中text_phone是根据TextGrid文件生成的
                                  第三个为去掉离群值后的 Frame-level pitch，是一个一维数组，长度为duration也就是帧数
                                  第四个为去掉离群值后的 Phone-level pitch，是一个一维数组，长度为duration也就是帧数
                                  第五个为去掉离群值后的 Frame-level energy，是一个一维数组，长度为duration也就是帧数
                                  第六个为去掉离群值后的 Phone-level energy，是一个一维数组，长度为duration也就是帧数
                                  第七个元素为  在帧级梅尔谱的帧数和音素级梅尔谱的帧数之间取最大值，用来告诉调用者这一条音频最终有多少帧。"""

            return (
                "|".join([basename, speaker, text_frame, raw_text, emotion]) if frame_out_exist else None,
                "|".join([basename, speaker, text_phone, raw_text, emotion]) if phone_out_exist else None,
                self.remove_outlier(pitch_frame) if frame_out_exist else None,
                self.remove_outlier(pitch_phone) if phone_out_exist else None,
                self.remove_outlier(energy_frame) if frame_out_exist else None,
                self.remove_outlier(energy_phone) if phone_out_exist else None,
                max(mel_spectrogram_frame.shape[1] if frame_out_exist else -1, mel_spectrogram_phone.shape[1] if phone_out_exist else -1)
            )

    def beta_binomial_prior_distribution(self, phoneme_count, mel_count, scaling_factor=1.0):
        P, M = phoneme_count, mel_count
        x = np.arange(0, P)
        mel_text_probs = []
        for i in range(1, M+1):
            a, b = scaling_factor*i, scaling_factor*(M+1-i)
            rv = betabinom(P, a, b)
            mel_i_prob = rv.pmf(x)
            mel_text_probs.append(mel_i_prob)
        return np.array(mel_text_probs)

    def get_alignment(self, tier):
        """函数返回的是根据某个音频的.TextGrid文件中的标签为 name = "phones"，得到的 phones, durations, start_time, end_time\n
        其中phones已经去掉了开头和结尾的静音标记（"sil"、"sp"、"spn"）的音素字符串列表；\n duration是整数列表，与 phone 中每个音素一一对应的帧数\n
        start是float，第一条非静音音素的起始时间（秒）\n
        end也是float，最后一条非静音音素的结束时间（秒）"""
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        """函数会把原始 values 中小于 lower 或大于 upper 的“离群”点去掉，返回一个只包含“正常”值的 NumPy 一维数组。"""
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)  #生成一个 布尔型（boolean）掩码数组,来标记所谓的“正常范围”内的点

        return values[normal_indices]   #“从数组 values 中，挑出那些 normal_indices 对应位置为 True 的元素，组成一个新的 1D 数组返回。”

    def normalize(self, in_dir, mean, std):
        """in_dir是preprocessed_data/DailyTalk/pitch_phone或pitch_frame或能量的音素，帧级目录\n
        返回min_value 和 max_value ，就是所有文件归一化后数据的最小值和最大值。"""
        max_value = np.finfo(np.float64).min  #max_value 被设为 float64 能表示的最小值（如 –1e308），用于后续取真正的最大值。
        min_value = np.finfo(np.float64).max  #min_value 被设为 float64 能表示的最大值（如 +1e308），用于后续取真正的最小值。
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            #np.load(filename) 读取每个 .npy 文件中的数组。减去给定的 mean、除以给定的 std，完成标准化：
            values = (np.load(filename) - mean) / std
            np.save(filename, values)  #将归一化后的数组重新写回到同名文件，原始数据被替换。
            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
