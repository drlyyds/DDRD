import json
import math
from typing import List, Any, Tuple
import os
import librosa
import numpy as np
#from sympy import false
from torch.utils.data import Dataset
import torch
from text import text_to_sequence
from utils.tools import get_variance_level, pad_1D, pad_2D, pad_3D

#针对于train调用dataset = Dataset("train_frame.txt", preprocess_config, model_config, train_config, sort=True, drop_last=True)
class Dataset(Dataset):
    #sort、drop_last：控制是否对数据做排序，和最后一个 batch 是否丢弃不满 batch_size 的样本。
    #filename="train_frame.txt"
    def __init__(
        self, filename, preprocess_config, model_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        """dataset_name =dailytalk"""
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        """preprocessed_path ="./preprocessed_data/DailyTalk"   """
        self.raw_path = preprocess_config["path"]["raw_path"]
        """raw_path ="./raw_data/DailyTalk"   """
        self.sub_dir_name = preprocess_config["path"]["sub_dir_name"]
        """sub_dir_name ="data"      """
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        """cleaners =["english_cleaners"]   """
        self.batch_size = train_config["optimizer"]["batch_size"]
        """batch_size=16 """
        self.learn_alignment = model_config["duration_modeling"]["learn_alignment"]
        """ learn_alignment =true  """
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'

        """load_spker_embed=flase，    计算方法：如果model_config["multi_speaker"]为false，直接返回load_spker_embed=false，如果model_config["multi_speaker"]为真，才会看后面的preprocess_config["preprocessing"]["speaker_embedder"] != 'none'，如果为真返回真，如果为假返回假   """
        self.load_emotion = model_config["multi_emotion"]
        """load_emotion=true    """
        self.history_type = model_config["history_encoder"]["type"]
        """history_type="Guo"   """
        self.text_emb_size = model_config["history_encoder"]["text_emb_size"]
        """text_emb_size=512"""
        self.max_history_len = model_config["history_encoder"]["max_history_len"]
        """max_history_len=10"""

        self.pitch_level_tag, self.energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)  #pitch_level_tag='frame'，energy_level_tag='frame'

        #basename，speaker，text，raw_text，emotion分别是五个字符串列表，存放train_frame.txt每个音频的信息
        self.basename, self.speaker, self.text, self.raw_text, self.emotion = self.process_meta(
            filename
        )

        self.basename_to_id = dict((v, k) for k, v in enumerate(self.basename))
        """basename_to_id={"0_1_d23":0,....}类似于这样格式的字典，其中左边的键是按照train_frame.txt中的顺序排列的basename，值就是从开始的数字"""

        #speakers.json和emotions.json中的内容保存为字典
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
            self.emotion_map = json.load(f)
        self.sort = sort
        """sort=true"""
        self.drop_last = drop_last
        """drop_last=true"""

    def __len__(self):
        """返回的是train_frame中音素序列的个数，也就是音频数量(一个对话必定有多个音频，一问一答算两个音频)"""
        return len(self.text)

    def __getitem__(self, idx):
        """输入id，返回的是sample \n
        sample = {
            "id": basename,\n
            "speaker": speaker_id,\n
            "text": phone,\n
            "raw_text": raw_text,\n
            "mel": mel,\n
            "pitch": pitch,\n
            "energy": energy,\n
            "duration": duration,  #为None  \n
            "attn_prior": attn_prior,\n
            "spker_embed": spker_embed,  #为None \n
            "emotion": emotion_id,\n
            "history": history,\n
            其中history = {
                    "text_emb": text_emb,\n
                    "history_len": history_len,\n
                    "history_text_emb": history_text_emb,\n
                    "history_speaker": history_speaker,\n
                }
        }"""

        #这里的basename，speaker都是单个音频的basename和speaker，
        basename = self.basename[idx]  #basename[idx]中的basename是训练集所有音频构成的列表
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        emotion_id = self.emotion_map[self.emotion[idx]] if self.load_emotion else None
        raw_text = self.raw_text[idx]
        #self.text[idx]是形如这样的"{Y EH1 S sp W IY1}"，然后根据传入英文清洗，会全部小写，去掉{},然后音素之间的空格只保留一个，-》》清洗后变成"y eh1 s sp w iy1"
        #text_to_sequence会把音素映射为int整数，相当于序号列表
        #np.array(...) 会返回一个 numpy.ndarray 对象，就是 NumPy 里最核心的多维数组类型。
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        """phone其实是一维 ndarray，也就是numpy里的一维数组，其中的元素是当前音频的音素映射为的id"""
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel_{}".format(self.pitch_level_tag),
            "{}-mel-{}.npy".format(speaker, basename),
        )
        """mel_path等于这个idx对应的那个音频的，preprocessed_data/DailyTalk/mel_frame下的.npy文件路径"""
        mel = np.load(mel_path)
        """mel等于这个idx对应的那个音频的，mel_frame目录下,帧级mel特征，对应的那个numpy类型的二维数组(duration,80)"""
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch_{}".format(self.pitch_level_tag),
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        """pitch等于这个idx对应的那个音频的，pitch_frame目录下,帧级pitch特征，对应的那个numpy类型的一维数组(duration,)"""
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy_{}".format(self.energy_level_tag),
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        """energy等于这个idx对应的那个音频的，energy_frame目录下,帧级energy特征，对应的那个numpy类型的一维数组(duration,)"""

        #如果对齐学习=true，就会用到先验注意力矩阵
        if self.learn_alignment:
            attn_prior_path = os.path.join(
                self.preprocessed_path,
                "attn_prior",
                "{}-attn_prior-{}.npy".format(speaker, basename),
            )
            attn_prior = np.load(attn_prior_path)
            """attn_prior等于这个idx对应的那个音频的，attn_prior目录下,帧级先验注意力矩阵，对应的那个numpy类型的二维数组(N,T)N代表音素个数，T代表duration也就是帧长"""
            duration = None
        # 反之如果对齐学习=false，就会用到preprocessed_data/DailyTalk/duration下的
        else:
            duration_path = os.path.join(
                self.preprocessed_path,
                "duration",
                "{}-duration-{}.npy".format(speaker, basename),
            )
            duration = np.load(duration_path)
            """duration等于这个idx对应的那个音频的，duration目录下,音素级帧长，对应的那个numpy类型的一维数组(duration,)，每个元素为对应音素的帧长"""
            attn_prior = None


        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None
        """spker_embed=none"""

        # History
        dialog = basename.split("_")[2].strip("d")
        """dialog=当前音频的所属对话id,是个字符串"""
        turn = int(basename.split("_")[0])
        """turn=当前音频在所属对话中的id，int类型"""

        history_len = min(self.max_history_len, turn)
        """history_len为10和（当前音频在所属对话中的id）中的最小值"""
        history_text = list()
        """history_texts是一个列表，里面存放的是历史音频的音素序列id"""
        history_text_emb = list()
        """history_text_emb为一个空list，存放的是历史音频的文本嵌入"""
        history_wav=list()
        """history_wav是一个列表,里面存放的是历史音频采样得到的numpy类型的一维浮点数组，这个列表中的元素个数为历史音频的个数"""
        history_text_len = list()
        history_pitch = list()
        history_energy = list()
        history_duration = list()
        history_emotion = list()
        history_speaker = list()
        """history_speaker为一个空list，存放的是历史音频的说话人id"""
        history_mel_len = list()
        history = None
        history_phone_seq=[]
        history_phone_seq_mask=[]
        history_wav=[]
        history_wav_calp=[]
        history_wav_mask=[]
        history_txt_calp=[]
        if self.history_type != "none":
            #第一个history_basenames是一个排序后的，某个对话id的所有音频文件名构成的列表，排序规则是按音频在对话中的id，形如[0_1_d0，1_0_d0，2_1_d0....]
            history_basenames = sorted([tg_path.replace(".wav", "") for tg_path in os.listdir(os.path.join(self.raw_path, self.sub_dir_name, f"{dialog}")) if ".wav" in tg_path], key=lambda x:int(x.split("_")[0]))
            """history_basenames是一个列表，里面的元素当前音频的历史语音的basename，"""   #上面这个history_basenames是一个列表，不过是所属对话的所有音频的history_basenames

            history_basenames = history_basenames[:turn][-history_len:]  #先切片当前音频之前的所有历史语音，再从后向前得到符合要求数量(10和当前话语id取最小)的历史语音
            """history_basenames是一个列表，里面的元素当前音频的历史语音的basename，前面这句话是第二个history_basenames"""

            if self.history_type == "Guo":
                text_emb_path = os.path.join(
                    self.preprocessed_path,
                    "text_emb",
                    "{}-text_emb-{}.npy".format(speaker, basename),
                )
                text_emb = np.load(text_emb_path)
                """text_emb是当前音频所对应的文本的嵌入向量"""

                max_phone_len=1  #最大音素序列长度
                max_n_samples=1   #最大音频采样点个数
                """某个音频的历史音频的最大音素序列个数"""
                for i, h_basename in enumerate(history_basenames):
                    h_idx = int(self.basename_to_id[h_basename])
                    """求得当前音频的当前历史音频在训练集train_frame.txt的行数或者是id"""

                    his_phone=np.array(text_to_sequence(self.text[h_idx], self.cleaners))
                    "求得当前音频的当前历史音频的音素序列id，是一个一维数组"
                    if(his_phone.shape[0] > max_phone_len):
                        max_phone_len = his_phone.shape[0]
                    history_text.append(text_to_sequence(self.text[h_idx], self.cleaners))

                    ######加载音频为一维数组
                    wav_path=os.path.join("./data",dialog,"{}.wav".format(basename))
                    wav, _ = librosa.load(wav_path, sr=16000)
                    wav_clap,_=librosa.load(wav_path, sr=48000)
                    """求得当前音频的当前历史音频转化为的一维numpy数组，shape为(n_samples,)，n_samples为采样点的个数"""
                    if(wav.shape[0] > max_phone_len):
                        max_n_samples = wav.shape[0]
                    history_wav.append(wav)
                    history_wav_calp.append(wav_clap)
                    ######提取原始文本字符串
                    text_content = ""  # 初始化一个空字符串，用于存储文件内容
                    text_string_path=os.path.join("./raw_data/DailyTalk/data",dialog,"{}.lab".format(basename))
                    try:
                        # 'with open' 会在代码块执行完毕后自动关闭文件，非常安全
                        # 'r' 表示以“只读”模式打开文件
                        # 'encoding="utf-8"' 是非常重要的好习惯，可以避免因文件编码问题导致的乱码或错误
                        with open(text_string_path, 'r', encoding='utf-8') as f:
                            # f.read() 会一次性读取整个文件的所有内容并返回一个字符串
                            text_content = f.read()

                    except FileNotFoundError:
                        print(f"错误：找不到文件，请检查路径是否正确: {text_string_path}")
                        # 在这里你可以选择进行其他错误处理，比如跳过、记录日志等

                    except Exception as e:
                        print(f"读取文件时发生未知错误: {e}")
                        # 捕获其他可能的异常
                    history_txt_calp.append(text_content)


                    h_speaker = self.speaker[h_idx]
                    """求得当前音频的当前历史音频的说话人id"""
                    h_speaker_id = self.speaker_map[h_speaker] #因为这里speaker——map对应speaker.json文件，键值相等，所以和上面的h_speaker没区别
                    """求得当前音频的当前历史音频的说话人id"""
                    h_text_emb_path = os.path.join(
                        self.preprocessed_path,
                        "text_emb",
                        "{}-text_emb-{}.npy".format(h_speaker, h_basename),
                    )
                    h_text_emb = np.load(h_text_emb_path)
                    """h_text_emb为当前历史语音的文本嵌入"""
                    history_text_emb.append(h_text_emb)
                    history_speaker.append(h_speaker_id)


                    # Padding
                    #如果是history_len<预定的10并且是最后一个历史音频(也就是当前音频在对话中的上一个音频)的时候，执行pad_history函数，填充0，保证长度为10
                    if i == history_len-1 and history_len < self.max_history_len:
                        self.pad_history(
                            self.max_history_len-history_len,
                            history_text_emb=history_text_emb,
                            history_speaker=history_speaker,
                        )
                #padding
                history_phone_seq, history_phone_seq_mask = self.pad_history_text_with_mask(history_text,self.max_history_len,max_phone_len)
                history_wav, history_wav_mask = self.pad_history_wav_with_mask(history_wav, self.max_history_len,max_n_samples)

                #如果turn == 0，也就是对话中的第一句话，直接将history_text_emb和history_speaker这两个空list，填补10个0
                if turn == 0:
                    self.pad_history(
                        self.max_history_len,
                        history_text_emb=history_text_emb,
                        history_speaker=history_speaker,
                    )
                    history_phone_seq,history_phone_seq_mask=self.pad_history_text_with_mask(history_text, self.max_history_len, max_phone_len)
                    history_wav, history_wav_mask = self.pad_history_wav_with_mask(history_wav, self.max_history_len,max_n_samples)

                history = {
                    "text_emb": text_emb,
                    "history_len": history_len,
                    "history_text_emb": history_text_emb,
                    "history_speaker": history_speaker,
                    "history_phone_seq": history_phone_seq,
                    "history_phone_seq_mask": history_phone_seq_mask,
                    "history_wav": history_wav,
                    "history_wav_mask": history_wav_mask,
                    "history_wav_clap":history_wav_calp,
                    "history_txt_clap":history_txt_calp
                }

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,   #为None
            "attn_prior": attn_prior,
            "spker_embed": spker_embed,  #为None
            "emotion": emotion_id,
            "history": history,
        }
        """sample为一个字典"""

        return sample
    #self.pad_history(10-实际历史音频长度,history_text_emb=[历史音频文本嵌入集合],history_speaker=[历史音频说话人id集合])
    def pad_history(self,
            pad_size,
            history_text=None,
            history_text_emb=None,
            history_text_len=None,
            history_pitch=None,
            history_energy=None,
            history_duration=None,
            history_emotion=None,
            history_speaker=None,
            history_mel_len=None,
        ):
        """pad_history，传入参数为history_text_emb和history_speaker，作用是保证这两个列表长度为10，也就是预设的对话历史长度，如果本身不够就用0来填充"""
        for _ in range(pad_size):
            history_text.append(np.zeros(1, dtype=np.int64)) if history_text is not None else None
            history_text_emb.append(np.zeros(self.text_emb_size, dtype=np.float32)) if history_text_emb is not None else None
            history_text_len.append(0) if history_text_len is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_pitch.append(np.zeros(1, dtype=np.float64)) if history_pitch is not None else None
            history_energy.append(np.zeros(1, dtype=np.float32)) if history_energy is not None else None
            history_duration.append(np.zeros(1, dtype=np.float64)) if history_duration is not None else None
            history_emotion.append(0) if history_emotion is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_speaker.append(0) if history_speaker is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_mel_len.append(0) if history_mel_len is not None else None # meaningless zero padding, should be cut out by mask of history_len

    #将一个音频的历史音素序列padding到统一长度
    def pad_history_text_with_mask(self,
            history_text,
            max_history_len,
            max_phone_len,
            pad_id=0
    ):
        """
        将 history_text 中的每条序列 pad 到相同长度，并 pad 到固定行数，同时返回对应的 mask 矩阵。

        参数:
        - history_text: list of list of int, 实际的历史音素 ID 序列（长度 history_len ≤ max_history_len）
        - max_history_len: int, 最多的历史序列数（结果行数）
        - max_phone_len: int, 每行 pad 后的长度（结果列数）
        - pad_id: int, 用于填充的 ID（padding 部分填入此值）

        返回:
        - padded_array: np.ndarray，shape=(max_history_len, max_phone_len)
          pad 后的音素 ID 矩阵
        - mask: np.ndarray of bool，shape=(max_history_len, max_phone_len)
          True 表示有效音素位置，False 表示 padding
        """
        # 初始化全 pad_id 矩阵和全 False mask
        padded_array = np.full((max_history_len, max_phone_len), pad_id, dtype=int)
        mask = np.zeros((max_history_len, max_phone_len), dtype=bool)

        # 只填充实际存在的 history_text 行，其余保持 pad_id 和 False
        for i, seq in enumerate(history_text[:max_history_len]):
            length = min(len(seq), max_phone_len)
            padded_array[i, :length] = seq[:length]
            mask[i, :length] = True

        return padded_array, mask

    def pad_history_wav_with_mask(self,
            history_wav: list,
            max_history_len: int,
            max_n_samples: int,
            pad_value: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        将 history_wav 中的每条波形 pad 或截断到相同长度，并 pad 到固定行数，同时返回对应的 mask 矩阵。

        参数:
        - history_wav: list of np.ndarray, 每个元素是一维浮点波形数组
        - max_history_len: int, 结果的最大行数（历史条数）
        - max_n_samples: int, 每条序列 pad 或截断后的样本点数（列数）
        - pad_value: float, 用于填充的值，默认 0.0

        返回:
        - padded_array: np.ndarray, shape = (max_history_len, max_n_samples), dtype=float32
        - mask: np.ndarray of bool, same shape, True 表示有效波形位置，False 表示 padding
        """
        # 初始化全 pad_value 矩阵和全 False mask
        padded_array = np.full((max_history_len, max_n_samples), pad_value, dtype=np.float32)
        mask = np.zeros((max_history_len, max_n_samples), dtype=bool)

        # 填充已有的历史波形及 mask
        for i, wav in enumerate(history_wav[:max_history_len]):
            # wav: 一维 numpy 数组
            length = min(len(wav), max_n_samples)
            padded_array[i, :length] = wav[:length]
            mask[i, :length] = True

        return padded_array, mask

    #padding,一个batch的序列
    def pad_batch_sequences(self,
            batch_seq: List[np.ndarray],
            pad_value: Any = 0
    ) -> np.ndarray:
        """
        将一批 2D 数组按第二维度 pad 到相同长度，返回 3D 数组。

        参数:
        - batch_seq: List of np.ndarray, 每个元素形状 (H, L_i)，
          H 是固定的行数（如 10），L_i 是可变的列长度。
        - pad_value: 填充值，默认为 0，可用于数值或布尔掩码（False）。

        返回:
        - padded: np.ndarray, shape = (B, H, max_L)
          B = len(batch_seq), max_L = max(L_i)。
        """
        B = len(batch_seq)
        if B == 0:
            # 如果批量是空，返回空数组
            return np.empty((0, 0, 0), dtype=type(pad_value))

        # 假设所有 array 的第一维 H 相同
        H = batch_seq[0].shape[0]
        # 计算最大列长度
        max_L = max(arr.shape[1] for arr in batch_seq)
        dtype = batch_seq[0].dtype

        # 初始化全 pad_value 的 3D 数组
        padded = np.full((B, H, max_L), pad_value, dtype=dtype)

        # 填充每个样本
        for b, arr in enumerate(batch_seq):
            L_i = arr.shape[1]
            padded[b, :, :L_i] = arr

        return padded

    def process_meta(self, filename):
        """传入的参数是train_frame.txt
        返回值name,speaker,text,raw_text,emotion这些都是五个列表，train_frame.txt中每个音频的basename，说话人id，音素序列，原始文本，情感分别放在这五个列表中"""
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            emotion = []
            for line in f.readlines():
                if self.load_emotion:
                    n, s, t, r, e = line.strip("\n").split("|")
                else:
                    n, s, t, r, *_ = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                if self.load_emotion:
                    emotion.append(e)
            return name, speaker, text, raw_text, emotion

    #output.append(self.reprocess(data, idx))，其中idx是一个形状为 (batch_size,）的一维数组，data为一个 list，每项是 __getitem__ 返回的那个 sample字典
    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        """ids为一个list列表，里面是basename，根据传入的一个batch的所有样本id"""
        speakers = [data[idx]["speaker"] for idx in idxs]
        """speakers为一个list列表，里面是说话人id，根据传入的一个batch的所有样本id"""
        texts = [data[idx]["text"] for idx in idxs]
        """texts为一个list列表，里面是音频的音素序列对应的int整型列表，例如[[],[],.....]，，，根据传入的一个batch的所有样本id,最终为numpy类型的二维数组，并且每个一维数组长度相同（用0填充），(batchsize, 最大音素序列长度)"""
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        """raw_texts为一个字符串list列表，里面是音频文本，根据传入的一个batch的所有样本id"""
        mels = [data[idx]["mel"] for idx in idxs]
        """mels为一个list列表，里面每个元素是音频的mel特征，根据传入的一个batch的所有样本id，mels大致是这样的：[二维数组，二维数组，....]，二维数组形状为(duration,80),最终为numpy类型的三维数组，并且每个二维数组形状相同（用0填充），(batchsize, 最大帧长，80)"""
        pitches = [data[idx]["pitch"] for idx in idxs]
        """pitches为一个list列表，里面每个元素是音频的pitch特征，根据传入的一个batch的所有样本id，pitches大致是这样的：[一维数组，一维数组，....]，一维数组长度为duration,最终为numpy类型的二维数组，并且每个一维数组长度相同（用0填充），(batchsize, 最大帧长)"""
        energies = [data[idx]["energy"] for idx in idxs]
        """energies为一个list列表，里面每个元素是音频的energy特征，根据传入的一个batch的所有样本id，energies大致是这样的：[一维数组，一维数组，....]，一维数组长度为duration,最终为numpy类型的二维数组，并且每个一维数组长度相同（用0填充），(batchsize, 最大帧长)"""
        durations = [data[idx]["duration"] for idx in idxs] if not self.learn_alignment else None
        """durations为none"""
        attn_priors = [data[idx]["attn_prior"] for idx in idxs] if self.learn_alignment else None
        """attn_priors为一个list列表，里面每个元素是音频的先验注意力矩阵，根据传入的一个batch的所有样本id，attn_priors大致是这样的：[二维数组，二维数组，....]，二维数组形状为(N, duration)，行为音素个数,最终为numpy类型的三维数组，并且每个二维数组形状相同（用0填充），(batchsize, 最大音素序列长度，最大帧长)"""
        spker_embeds = np.concatenate(np.array([data[idx]["spker_embed"] for idx in idxs]), axis=0) \
            if self.load_spker_embed else None
        """spker_embeds为none"""

        emotions = np.array([data[idx]["emotion"] for idx in idxs]) if self.load_emotion else None
        """emotions为一个numpy类型的一维数组，里面是emotion的id，根据传入的一个batch的所有样本id"""

        text_lens = np.array([text.shape[0] for text in texts])
        """text_lens是numpy类型的一维数组，其中的每个元素是一个batch中音频的音素序列长度(音频的音素个数)"""
        mel_lens = np.array([mel.shape[0] for mel in mels])
        """mel_lens是numpy类型的一维数组，其中的每个元素是一个batch中音频的帧长"""

        #将列表转为numpy类型的数组
        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        if self.learn_alignment:
            attn_priors = pad_3D(attn_priors, len(idxs), max(text_lens), max(mel_lens))
        else:
            durations = pad_1D(durations)

        history_info = None
        if self.history_type != "none":
            if self.history_type == "Guo":
                text_embs = [data[idx]["history"]["text_emb"] for idx in idxs]
                """一个列表，存放这个batch中音频对应文本的嵌入，最终转为numpy数组(B,512)"""
                history_lens = [data[idx]["history"]["history_len"] for idx in idxs]
                """一个列表，存放这个batch中该音频要利用的历史语音的个数，最终转为numpy数组(B,)"""
                history_text_embs = [data[idx]["history"]["history_text_emb"] for idx in idxs]
                """一个列表，列表中的元素是一个子列表(长度为10)，子列表中的元素存放这个batch中该音频要利用的历史语音的文本嵌入，不足10用0向量嵌入，最终转为numpy数组(B, 10, 512)"""
                history_speakers = [data[idx]["history"]["history_speaker"] for idx in idxs]
                """一个列表，列表中的元素是一个子列表(长度为10)，子列表中的元素存放这个batch中该音频要利用的历史语音的说话人id，不足10用0向量嵌入，最终转为numpy数组(B, 10)"""
                history_phone_seq=  [data[idx]["history"]["history_phone_seq"] for idx in idxs]  #(B,10,T)
                history_phone_seq_mask = [data[idx]["history"]["history_phone_seq_mask"] for idx in idxs] #(B,10,T)
                history_wav = [data[idx]["history"]["history_wav"] for idx in idxs]  #(B,10,n_samples)
                history_wav_mask = [data[idx]["history"]["history_wav_mask"] for idx in idxs]   #(B,10,n_samples)
                history_wav_clap = [data[idx]["history"]["history_wav_clap"] for idx in idxs]
                history_txt_clap = [data[idx]["history"]["history_txt_clap"] for idx in idxs]

                history_phone_seq=self.pad_batch_sequences(history_phone_seq,0)
                history_phone_seq_mask=self.pad_batch_sequences(history_phone_seq_mask,False)
                history_wav=self.pad_batch_sequences(history_wav,0.0)
                history_wav_mask=self.pad_batch_sequences(history_wav_mask,False)

                #上述这些转为numpy类型的数组
                text_embs = np.array(text_embs)
                history_lens = np.array(history_lens)
                history_text_embs = np.array(history_text_embs)
                history_speakers = np.array(history_speakers)

                history_info = (
                    text_embs,
                    history_lens,
                    history_text_embs,
                    history_speakers,
                    history_phone_seq,
                    history_phone_seq_mask,
                    history_wav,
                    history_wav_mask,
                    history_wav_clap,
                    history_txt_clap
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                processed_elements = []
                for x in history_info:
                    # 检查 x 是否是可以直接转换为张量的类型
                    if isinstance(x, (torch.Tensor, np.ndarray)):
                        # 如果是，就执行转换和设备移动操作
                        tensor_x = x if torch.is_tensor(x) else torch.from_numpy(x)
                        processed_elements.append(tensor_x.to(device))
                    else:
                        # 如果是其他类型 (比如您的嵌套字符串列表 history_txt_clap)
                        # 那么就保持原样，直接添加到新列表中
                        processed_elements.append(x)
                history_info = tuple(processed_elements)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            attn_priors,
            spker_embeds,
            emotions,
            history_info,
        )

    def collate_fn(self, data):
        """输入参数data为一个 list，共有batchsize*4项，因为一开始是划组，每项是 __getitem__ 返回的那个 sample字典.......
        \n   data=[{}，{}，{}，{}，....]
        输出是一个列表[]，列表中的每一个元素是一个元组，(ids,raw_texts,speakers,texts,text_lens,max(text_lens),mels,mel_lens,max(mel_lens),pitches,energies,durations,attn_priors,spker_embeds,emotions,history_info,)这是一个batch的所有音频的信息，用一个元组表示,
        有几个batch就有几个这样的元组"""
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            """len_arr是一个numpy类型的一维数组，其中的每个元素是传入的data中的音频的音素序列的个数"""
            idx_arr = np.argsort(-len_arr)
            """idx_arr是一个numpy类型的一维数组，其中的每个元素索引，告诉你按照data中的音频的音素个数从大到小排序后，应该按哪些索引顺序去取。\n
            例如len_arr = np.array([10, 3, 7, 5])，那么idx_arr =np.array([0, 2, 3, 1])"""
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        """tail numpy类型的一维数组承载这些“剩余不满 batch_size 的索引。"""
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]  #相应地，现在的index_arr就是去除不满 batch_size 的索引后的结果
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()  #将idx_arr把它切成 batche数 × batch_size 的二维数组，并转为list的形式

        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


#用在合成,dataset = TextDataset(args.source, preprocess_config, model_config)
class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config, model_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"] #["english_cleaners"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"] #"./preprocessed_data/DailyTalk"
        self.raw_path = preprocess_config["path"]["raw_path"]  #"./raw_data/DailyTalk"
        self.sub_dir_name = preprocess_config["path"]["sub_dir_name"] #"data"
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        """load_spker_embed=false"""
        self.load_emotion = model_config["multi_emotion"]   #true
        self.history_type = model_config["history_encoder"]["type"]   #GUO
        self.text_emb_size = model_config["history_encoder"]["text_emb_size"]  #512
        self.max_history_len = model_config["history_encoder"]["max_history_len"]  #10

        self.basename, self.speaker, self.text, self.raw_text, self.emotion = self.process_meta(
            filepath
        )
        self.basename_to_id = dict((v, k) for k, v in enumerate(self.basename))
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
            self.emotion_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        emotion_id = self.emotion_map[self.emotion[idx]] if self.load_emotion else None
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None

        # History
        dialog = basename.split("_")[2].strip("d")
        """dialog=当前音频的所属对话id,是个字符串"""
        turn = int(basename.split("_")[0])
        """turn=当前音频在所属对话中的id，int类型"""

        history_len = min(self.max_history_len, turn)
        """history_len为10和（当前音频在所属对话中的id）中的最小值"""
        history_text = list()
        """history_texts是一个列表，里面存放的是历史音频的音素序列id"""
        history_text_emb = list()
        """history_text_emb为一个空list，存放的是历史音频的文本嵌入"""
        history_wav = list()
        """history_wav是一个列表,里面存放的是历史音频采样得到的numpy类型的一维浮点数组，这个列表中的元素个数为历史音频的个数"""
        history_text_len = list()
        history_pitch = list()
        history_energy = list()
        history_duration = list()
        history_emotion = list()
        history_speaker = list()
        """history_speaker为一个空list，存放的是历史音频的说话人id"""
        history_mel_len = list()
        history = None
        history_phone_seq = []
        history_phone_seq_mask = []
        history_wav = []
        history_wav_calp = []
        history_wav_mask = []
        history_txt_calp = []
        if self.history_type != "none":
            # 第一个history_basenames是一个排序后的，某个对话id的所有音频文件名构成的列表，排序规则是按音频在对话中的id，形如[0_1_d0，1_0_d0，2_1_d0....]
            history_basenames = sorted([tg_path.replace(".wav", "") for tg_path in
                                        os.listdir(os.path.join(self.raw_path, self.sub_dir_name, f"{dialog}")) if
                                        ".wav" in tg_path], key=lambda x: int(x.split("_")[0]))
            """history_basenames是一个列表，里面的元素当前音频的历史语音的basename，"""  # 上面这个history_basenames是一个列表，不过是所属对话的所有音频的history_basenames

            history_basenames = history_basenames[:turn][
                                -history_len:]  # 先切片当前音频之前的所有历史语音，再从后向前得到符合要求数量(10和当前话语id取最小)的历史语音
            """history_basenames是一个列表，里面的元素当前音频的历史语音的basename，前面这句话是第二个history_basenames"""

            if self.history_type == "Guo":
                text_emb_path = os.path.join(
                    self.preprocessed_path,
                    "text_emb",
                    "{}-text_emb-{}.npy".format(speaker, basename),
                )
                text_emb = np.load(text_emb_path)
                """text_emb是当前音频所对应的文本的嵌入向量"""

                max_phone_len = 1  # 最大音素序列长度
                max_n_samples = 1  # 最大音频采样点个数
                """某个音频的历史音频的最大音素序列个数"""
                for i, h_basename in enumerate(history_basenames):
                    h_idx = int(self.basename_to_id[h_basename])
                    """求得当前音频的当前历史音频在训练集train_frame.txt的行数或者是id"""

                    his_phone = np.array(text_to_sequence(self.text[h_idx], self.cleaners))
                    "求得当前音频的当前历史音频的音素序列id，是一个一维数组"
                    if (his_phone.shape[0] > max_phone_len):
                        max_phone_len = his_phone.shape[0]
                    history_text.append(text_to_sequence(self.text[h_idx], self.cleaners))

                    ######加载音频为一维数组
                    wav_path = os.path.join("./data", dialog, "{}.wav".format(basename))
                    wav, _ = librosa.load(wav_path, sr=16000)
                    wav_clap, _ = librosa.load(wav_path, sr=48000)
                    """求得当前音频的当前历史音频转化为的一维numpy数组，shape为(n_samples,)，n_samples为采样点的个数"""
                    if (wav.shape[0] > max_phone_len):
                        max_n_samples = wav.shape[0]
                    history_wav.append(wav)
                    history_wav_calp.append(wav_clap)
                    ######提取原始文本字符串
                    text_content = ""  # 初始化一个空字符串，用于存储文件内容
                    text_string_path = os.path.join("./raw_data/DailyTalk/data", dialog, "{}.lab".format(basename))
                    try:
                        # 'with open' 会在代码块执行完毕后自动关闭文件，非常安全
                        # 'r' 表示以“只读”模式打开文件
                        # 'encoding="utf-8"' 是非常重要的好习惯，可以避免因文件编码问题导致的乱码或错误
                        with open(text_string_path, 'r', encoding='utf-8') as f:
                            # f.read() 会一次性读取整个文件的所有内容并返回一个字符串
                            text_content = f.read()

                    except FileNotFoundError:
                        print(f"错误：找不到文件，请检查路径是否正确: {text_string_path}")
                        # 在这里你可以选择进行其他错误处理，比如跳过、记录日志等

                    except Exception as e:
                        print(f"读取文件时发生未知错误: {e}")
                        # 捕获其他可能的异常
                    history_txt_calp.append(text_content)

                    h_speaker = self.speaker[h_idx]
                    """求得当前音频的当前历史音频的说话人id"""
                    h_speaker_id = self.speaker_map[
                        h_speaker]  # 因为这里speaker——map对应speaker.json文件，键值相等，所以和上面的h_speaker没区别
                    """求得当前音频的当前历史音频的说话人id"""
                    h_text_emb_path = os.path.join(
                        self.preprocessed_path,
                        "text_emb",
                        "{}-text_emb-{}.npy".format(h_speaker, h_basename),
                    )
                    h_text_emb = np.load(h_text_emb_path)
                    """h_text_emb为当前历史语音的文本嵌入"""
                    history_text_emb.append(h_text_emb)
                    history_speaker.append(h_speaker_id)

                    # Padding
                    # 如果是history_len<预定的10并且是最后一个历史音频(也就是当前音频在对话中的上一个音频)的时候，执行pad_history函数，填充0，保证长度为10
                    if i == history_len - 1 and history_len < self.max_history_len:
                        self.pad_history(
                            self.max_history_len - history_len,
                            history_text_emb=history_text_emb,
                            history_speaker=history_speaker,
                        )
                # padding
                history_phone_seq, history_phone_seq_mask = self.pad_history_text_with_mask(history_text,
                                                                                            self.max_history_len,
                                                                                            max_phone_len)
                history_wav, history_wav_mask = self.pad_history_wav_with_mask(history_wav, self.max_history_len,
                                                                               max_n_samples)

                # 如果turn == 0，也就是对话中的第一句话，直接将history_text_emb和history_speaker这两个空list，填补10个0
                if turn == 0:
                    self.pad_history(
                        self.max_history_len,
                        history_text_emb=history_text_emb,
                        history_speaker=history_speaker,
                    )
                    history_phone_seq, history_phone_seq_mask = self.pad_history_text_with_mask(history_text,
                                                                                                self.max_history_len,
                                                                                                max_phone_len)
                    history_wav, history_wav_mask = self.pad_history_wav_with_mask(history_wav, self.max_history_len,
                                                                                   max_n_samples)

                history = {
                    "text_emb": text_emb,
                    "history_len": history_len,
                    "history_text_emb": history_text_emb,
                    "history_speaker": history_speaker,
                    "history_phone_seq": history_phone_seq,
                    "history_phone_seq_mask": history_phone_seq_mask,
                    "history_wav": history_wav,
                    "history_wav_mask": history_wav_mask,
                    "history_wav_clap": history_wav_calp,
                    "history_txt_clap": history_txt_calp
                }

        return (basename, speaker_id, phone, raw_text, spker_embed, emotion_id, history)

    def pad_history(self,
            pad_size,
            history_text=None,
            history_text_emb=None,
            history_text_len=None,
            history_pitch=None,
            history_energy=None,
            history_duration=None,
            history_emotion=None,
            history_speaker=None,
            history_mel_len=None,
        ):
        for _ in range(pad_size):
            history_text.append(np.zeros(1, dtype=np.int64)) if history_text is not None else None
            history_text_emb.append(np.zeros(self.text_emb_size, dtype=np.float32)) if history_text_emb is not None else None
            history_text_len.append(0) if history_text_len is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_pitch.append(np.zeros(1, dtype=np.float64)) if history_pitch is not None else None
            history_energy.append(np.zeros(1, dtype=np.float32)) if history_energy is not None else None
            history_duration.append(np.zeros(1, dtype=np.float64)) if history_duration is not None else None
            history_emotion.append(0) if history_emotion is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_speaker.append(0) if history_speaker is not None else None # meaningless zero padding, should be cut out by mask of history_len
            history_mel_len.append(0) if history_mel_len is not None else None # meaningless zero padding, should be cut out by mask of history_len

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            emotion = []
            for line in f.readlines():
                if self.load_emotion:
                    n, s, t, r, e = line.strip("\n").split("|")
                else:
                    n, s, t, r, *_ = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                if self.load_emotion:
                    emotion.append(e)
            return name, speaker, text, raw_text, emotion

    def collate_fn(self, data):
        """传入的是8个样本的元组组成的list？，元组形如(basename, speaker_id, phone, raw_text, spker_embed, emotion_id, history)"""
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        spker_embeds = np.concatenate(np.array([d[4] for d in data]), axis=0) \
            if self.load_spker_embed else None
        """spker_embeds=none"""
        emotions = np.array([d[5] for d in data]) if self.load_emotion else None

        texts = pad_1D(texts)

        history_info = None
        if self.history_type != "none":
            if self.history_type == "Guo":
                text_embs = [d[6]["text_emb"] for d in data]
                history_lens = [d[6]["history_len"] for d in data]
                history_text_embs = [d[6]["history_text_emb"] for d in data]
                history_speakers = [d[6]["history_speaker"] for d in data]
                history_phone_seq = [d[6]["history_phone_seq"] for d in data]  # (B,10,T)
                history_phone_seq_mask = [d[6]["history_phone_seq_mask"] for d in data]  # (B,10,T)
                history_wav = [d[6]["history_wav"] for d in data]  # (B,10,n_samples)
                history_wav_mask = [d[6]["history_wav_mask"] for d in data]  # (B,10,n_samples)
                history_wav_clap = [d[6]["history_wav_clap"] for d in data]
                history_txt_clap = [d[6]["history_txt_clap"] for d in data]

                history_phone_seq = self.pad_batch_sequences(history_phone_seq, 0)
                history_phone_seq_mask = self.pad_batch_sequences(history_phone_seq_mask, False)
                history_wav = self.pad_batch_sequences(history_wav, 0.0)
                history_wav_mask = self.pad_batch_sequences(history_wav_mask, False)

                text_embs = np.array(text_embs)
                history_lens = np.array(history_lens)
                history_text_embs = np.array(history_text_embs)
                history_speakers = np.array(history_speakers)
                history_info = (
                    text_embs,
                    history_lens,
                    history_text_embs,
                    history_speakers,
                    history_phone_seq,
                    history_phone_seq_mask,
                    history_wav,
                    history_wav_mask,
                    history_wav_clap
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                processed_elements = []
                for x in history_info:
                    # 检查 x 是否是可以直接转换为张量的类型
                    if isinstance(x, (torch.Tensor, np.ndarray)):
                        # 如果是，就执行转换和设备移动操作
                        tensor_x = x if torch.is_tensor(x) else torch.from_numpy(x)
                        processed_elements.append(tensor_x.to(device))
                    else:
                        # 如果是其他类型 (比如您的嵌套字符串列表 history_txt_clap)
                        # 那么就保持原样，直接添加到新列表中
                        processed_elements.append(x)
                history_info = tuple(processed_elements)

        #在python里return a, b, c等价于return (a, b, c)，无需写上那对括号，逗号就已经把它们打包成了一个 tuple
        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embeds, emotions, history_info
