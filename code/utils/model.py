import os
import json

import torch
import numpy as np

import hifigan
from model.CompTransTTS import CompTransTTS
from model.optimizer import ScheduledOptim

#model, optimizer = get_model(args, configs, device, train=True)  #device = torch.device('cuda:{0}')
def get_model(args, configs, device, train=False):
    """返回 model（model是CompTransTTS类的对象）, scheduled_optim(ScheduledOptim类的对象)"""
    (preprocess_config, model_config, train_config) = configs

    model = CompTransTTS(preprocess_config, model_config, train_config).to(device)  #把模型参数放在Gpu上

    #args.restore_step 是一个整数（比如 325000），指想要加载的训练迭代步数。，也就是从某一个迭代步数开始训练
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device)  #把所有保存的张量（torch.Tensor）都直接分配到你指定的 device上
        # ckpt_path 是一个指向 .pth.tar、.pt 等文件的字符串(路径)，这是你之前用 torch.save(...) 写出的「检查点」文件。
        """此时 ckpt 是一个 dict，比如:
            {
                "model":     OrderedDict([...]),  # 所有张量都在 cuda:0
                "optimizer": OrderedDict([...]),  # 也都在 cuda:0
                "step":      325000,
                "loss":      1.2345,
            }"""
        #把从磁盘加载出来的参数字典（ckpt["model"]，一个 OrderedDict）“注入”到你的 model 对象里去——也就是把保存好的每一层权重、偏置、缓冲区全部恢复到当前模型。
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])   #如果是第 N 步断点续训，需要将优化器的状态同样恢复，因为学习率是动态改变的
        model.train()  #切换模型到训练模式
        return model, scheduled_optim


    #准备推理/验证阶段的模型状态，切换到“评估模式”，此时dropout不会起作用
    model.eval()
    model.requires_grad_ = False   #模型里所有参数的 requires_grad 属性都设为 False。保证参数不会更新
    return model


def get_param_num(model):
    """这个函数就是用来统计整个 model（通常是一个继承自 nn.Module 的网络）的可训练参数总数。"""

    #model.parameters() 会返回一个生成器，依次给出模型中所有的 Parameter 对象（nn.Parameter）（包括权重、偏置等）
    #param.numel()：对于每个参数张量，返回它包含的元素个数（行×列×通道…）。
    #sum(...)：把所有这些元素个数加在一起，就得到了模型里所有参数的总量。
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]  #universal

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)  #先读本地的 hifigan/config.json，得到模型超参（卷积层数、通道数等），并转换成 AttrDict 方便用属性方式访问。
        vocoder = hifigan.Generator(config)  #构建 HiFi‑GAN 的生成器网络结构
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar", map_location=device)
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar", map_location=device)  #反序列化得到字典对象
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval() #禁用 Dropout、BatchNorm 的训练行为
        vocoder.remove_weight_norm() #移除训练时用的权重归一化钩子，加速推理；
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
