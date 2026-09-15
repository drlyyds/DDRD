import os
import json
import yaml

import torch
import torch.nn.functional as F
from torch.cuda import amp
import numpy as np
import matplotlib
matplotlib.use("Agg")
from scipy.io import wavfile
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

#dataset="DailyTalk"
def get_configs_of(dataset):
    config_dir = os.path.join("./config", dataset)
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "r",encoding='utf-8'), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        os.path.join(config_dir, "model.yaml"), "r",encoding='utf-8'), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        os.path.join(config_dir, "train.yaml"), "r",encoding='utf-8'), Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config


def get_variance_level(preprocess_config, model_config, data_loading=True):
    """
    输入：两个配置文件preprocess_config, model_config，默认data_loading这个参数=True\n
    输出：return pitch_level_tag, energy_level_tag, pitch_feature_level, energy_feature_level\n
    pitch_level_tag(实际返回的是frame，energy也是)：字符串 "phone" 或 "frame"，用来告诉下游pitch 是哪一级别的。xxxx_feature_level(实际是phone，音素级):直接把preprocess.yaml配种文件中的配置原样返回，方便后续逻辑按照“特征级别”来加载或计算。\n
    对于xxxx_level_tag只有两件事都满足：1.不做无监督对齐（learn_alignment=False），2.且配置中提取的是音素级（feature_level == "phoneme_level"）。才能用音素级。其他情况都用frame，帧级

    """
    learn_alignment = model_config["duration_modeling"]["learn_alignment"] if data_loading else False    #如果在data_loading=true，也就是数据加载阶段，learn_alignment才等于配置文件中的值，否则永远为false
    pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]      #配置文件中使用音素级
    energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]    #配置文件中使用音素级
    assert pitch_feature_level in ["frame_level", "phoneme_level"]
    assert energy_feature_level in ["frame_level", "phoneme_level"]
    pitch_level_tag = "phone" if (not learn_alignment and pitch_feature_level == "phoneme_level") else "frame"
    energy_level_tag = "phone" if (not learn_alignment and energy_feature_level == "phoneme_level") else "frame"
    return pitch_level_tag, energy_level_tag, pitch_feature_level, energy_feature_level

#[get_phoneme_level_pitch(dur[:len], var) for dur, len, var in zip(duration.int().cpu().numpy(), src_len.cpu().numpy(), pitch_frame.cpu().numpy())],在cpu上执行的
def get_phoneme_level_pitch(duration, pitch):
    """传入参数：duration为一维numpy数组，长度为音频的实际音素序列长度，每个元素为音素的帧长；pitch为一维numpy数组，长度为D也就是最大帧长，每个元素为某帧的pitch；\n
    输出为：pitch，为一维numpy数组，长度为音频的实际音素序列长度，每个元素为对应音素的音素级pitch就是(音素对应的)插值过的帧级pitch取平均   """
    # perform linear interpolation
    nonzero_ids = np.where(pitch != 0)[0] #nonzero_ids 的类型就是 ndarray，形状 (K,)，其中K = 非零元素的数量(也就是实际帧长-静音帧长)，里面元素是非零id
    interp_fn = interp1d(
        nonzero_ids,
        pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        bounds_error=False,
    )  #以这些索引和对应的 pitch 值为控制点，建立一个一维线性插值函数 interp_fn。
    pitch = interp_fn(np.arange(0, len(pitch)))  #生成一个长度和原来一样、但所有位置都被插值过的新 pitch 序列(不再有0，而是被平滑填满了)

    # Phoneme-level average 音素级pitch就是(音素对应的)插值过的帧级pitch取平均
    pos = 0
    for i, d in enumerate(duration):
        if d > 0:
            pitch[i] = np.mean(pitch[pos : pos + d])
        else:
            pitch[i] = 0
        pos += d
    pitch = pitch[: len(duration)]
    return pitch


def get_phoneme_level_energy(duration, energy):
    # Phoneme-level average，和上面类似
    pos = 0
    for i, d in enumerate(duration):
        if d > 0:
            energy[i] = np.mean(energy[pos : pos + d])
        else:
            energy[i] = 0
        pos += d
    energy = energy[: len(duration)]
    return energy


def to_device(data, device):
    if len(data) == 16:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            attn_priors,
            spker_embeds,
            emotions,
            history_info,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        if durations is not None:
            durations = torch.from_numpy(durations).long().to(device)
        if attn_priors is not None:
            attn_priors = torch.from_numpy(attn_priors).float().to(device)
        if spker_embeds is not None:
            spker_embeds = torch.from_numpy(spker_embeds).float().to(device)
        if emotions is not None:
            emotions = torch.from_numpy(emotions).long().to(device)
        if history_info is not None:
            if len(history_info) == 4: # "Guo"
                (
                    text_embs,
                    history_lens,
                    history_text_embs,
                    history_speakers,
                ) = history_info

                text_embs = torch.from_numpy(text_embs).float().to(device)
                history_lens = torch.from_numpy(history_lens).to(device)
                history_text_embs = torch.from_numpy(history_text_embs).float().to(device)
                history_speakers = torch.from_numpy(history_speakers).long().to(device)

                history_info = (
                    text_embs,
                    history_lens,
                    history_text_embs,
                    history_speakers,
                )

        return [
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            attn_priors,
            spker_embeds,
            emotions,
            history_info,
        ]

    if len(data) == 9:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len, spker_embeds, emotions, history_info) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        if spker_embeds is not None:
            spker_embeds = torch.from_numpy(spker_embeds).float().to(device)
        if emotions is not None:
            emotions = torch.from_numpy(emotions).long().to(device)
        if history_info is not None:
            if len(history_info) == 4: # "Guo"
                (
                    text_embs,
                    history_lens,
                    history_text_embs,
                    history_speakers,
                ) = history_info

                text_embs = torch.from_numpy(text_embs).float().to(device)
                history_lens = torch.from_numpy(history_lens).to(device)
                history_text_embs = torch.from_numpy(history_text_embs).float().to(device)
                history_speakers = torch.from_numpy(history_speakers).long().to(device)

                history_info = (
                    text_embs,
                    history_lens,
                    history_text_embs,
                    history_speakers,
                )

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len, spker_embeds, emotions, history_info)


def log(
    logger, step=None, losses=None, fig=None, img=None, audio=None, sampling_rate=22050, tag=""
):
    """1.画出所有损失的曲线，横轴为损失值，纵坐标为某个损失值，一共至少8张图\n
     2. 记录 Matplotlib 图表\n
     3.记录静态图像
     4.记录可播放音频"""
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        for k, v in losses[5].items():
            logger.add_scalar("Loss/{}_loss".format(k), v, step)
        logger.add_scalar("Loss/ctc_loss", losses[6], step)
        logger.add_scalar("Loss/bin_loss", losses[7], step)
        logger.add_scalar("Loss/decoupling_loss", losses[8], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if img is not None:
        logger.add_image(tag, img, dataformats='HWC')

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),  #audio是一个长度为L（L是采样点的个数）的一维数组，然后/max是为了[-1.1]
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    """这个函数的作用是根据每条样本的有效长度，返回一个形状为 (B, X)x为max_len ，的布尔掩码张量 mask，用来屏蔽掉那些“填充”出来的无效位置。\n
        mask中true表示padding，false表示有效"""
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)  #ids的形状是[B,max_len]，ids[b, i] == i，每一行都是从 0 数到 max_len-1。
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)  #mask里的元素是true和false

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)

#train.py中使用的
def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):
    """合成一条语音"""
    learn_alignment = model_config["duration_modeling"]["learn_alignment"]
    pitch_level_tag, energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)
    basename = targets[0][0]
    src_len = predictions[8][0].item()
    mel_len = predictions[9][0].item()
    mel_target = targets[6][0, :mel_len].float().detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].float().detach().transpose(0, 1)
    duration = predictions[5][0, :src_len].int().detach().cpu().numpy()

    fig_attn = None
    if learn_alignment:
        attn_prior, attn_soft, attn_hard, attn_hard_dur, attn_logprob = targets[12], *predictions[10]
        attn_prior = attn_prior[0, :src_len, :mel_len].squeeze().detach().cpu().numpy() # text_len x mel_len
        attn_soft = attn_soft[0, 0, :mel_len, :src_len].detach().cpu().transpose(0, 1).numpy() # text_len x mel_len
        attn_hard = attn_hard[0, 0, :mel_len, :src_len].detach().cpu().transpose(0, 1).numpy() # text_len x mel_len
        fig_attn = plot_alignment(
            [
                attn_soft,
                attn_hard,
                attn_prior,
            ],
            ["Soft Attention", "Hard Attention", "Prior"]
        )

    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = targets[9][0, :src_len].float().detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = targets[9][0, :mel_len].float().detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets[10][0, :src_len].float().detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets[10][0, :mel_len].float().detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats[f"pitch_{pitch_level_tag}"] + stats[f"energy_{energy_level_tag}"][:2] # Should follow the level at data loading time.

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, fig_attn, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path, args):

    multi_speaker = model_config["multi_speaker"]
    multi_emotion = model_config["multi_emotion"]
    history_type = model_config["history_encoder"]["type"]
    emotion_tag = ("_" + args.emotion_id) if multi_emotion else ""
    learn_alignment = model_config["duration_modeling"]["learn_alignment"]
    pitch_level_tag, energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)
    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[5][i, :src_len].int().detach().cpu().numpy()
        attn_soft = attn_hard = None

        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats[f"pitch_{pitch_level_tag}"] + stats[f"energy_{energy_level_tag}"][:2] # Should follow the level at data loading time.

        if history_type == 'none':
            fig_save_dir = os.path.join(
                path, str(args.restore_step), "{}_{}{}.png".format(basename, args.speaker_id, emotion_tag)\
                    if multi_speaker and args.mode == "single" else "{}.png".format(basename))
        else:
            os.makedirs((os.path.join(
                path, str(args.restore_step), basename.split("_")[-1].strip("d"))), exist_ok=True)
            fig_save_dir = os.path.join(
                path, str(args.restore_step), basename.split("_")[-1].strip("d"), "{}.png".format(basename))
        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
            save_dir=fig_save_dir,
        )

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        if history_type == 'none':
            wav_save_dir = os.path.join(
                path, str(args.restore_step), "{}_{}{}.wav".format(basename, args.speaker_id, emotion_tag)\
                    if multi_speaker and args.mode == "single" else "{}.wav".format(basename))
        else:
            os.makedirs((os.path.join(
                path, str(args.restore_step), basename.split("_")[-1].strip("d"))), exist_ok=True)
            wav_save_dir = os.path.join(
                path, str(args.restore_step), basename.split("_")[-1].strip("d"), "{}.wav".format(basename))
        wavfile.write(wav_save_dir,sampling_rate, wav)


def plot_mel(data, stats, titles, save_dir=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato", linewidth=.7)
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet", linewidth=.7)
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.close()
    return data


# def plot_single_alignment(alignment, info=None, save_dir=None):
#     fig, ax = plt.subplots(figsize=(6, 4))
#     im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
#     fig.colorbar(im, ax=ax)
#     xlabel = 'Decoder timestep'
#     if info is not None:
#         xlabel += '\n\n' + info
#     plt.xlabel(xlabel)
#     plt.ylabel('Encoder timestep')
#     plt.tight_layout()

#     fig.canvas.draw()
#     data = save_figure_to_numpy(fig)
#     if save_dir is not None:
#         plt.savefig(save_dir)
#     plt.close()
#     return data


def plot_alignment(data, titles=None, save_dir=None):
    fig, axes = plt.subplots(len(data), 1, figsize=[6,4],dpi=300)
    plt.subplots_adjust(top = 0.9, bottom = 0.1, right = 0.95, left = 0.05)
    if titles is None:
        titles = [None for i in range(len(data))]

    for i in range(len(data)):
        im = data[i]
        axes[i].imshow(im, origin='lower')
        axes[i].set_xlabel('Audio timestep')
        axes[i].set_ylabel('Text timestep')
        axes[i].set_ylim(0, im.shape[0])
        axes[i].set_xlim(0, im.shape[1])
        axes[i].set_title(titles[i], fontsize='medium')
        axes[i].tick_params(labelsize='x-small')
        axes[i].set_anchor('W')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.close()
    return data


def plot_embedding(out_dir, embedding, embedding_speaker_id, gender_dict, filename='embedding.png'):
    colors = 'r','b'
    labels = 'Female','Male'

    data_x = embedding
    data_y = np.array([gender_dict[spk_id] == 'M' for spk_id in embedding_speaker_id], dtype=np.int)
    tsne_model = TSNE(n_components=2, random_state=0, init='random')
    tsne_all_data = tsne_model.fit_transform(data_x)
    tsne_all_y_data = data_y

    plt.figure(figsize=(10,10))
    for i, (c, label) in enumerate(zip(colors, labels)):
        plt.scatter(tsne_all_data[tsne_all_y_data==i,0], tsne_all_data[tsne_all_y_data==i,1], c=c, label=label, alpha=0.5)

    plt.grid(True)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def pad_1D(inputs, PAD=0, maxlen=None):
    """将一组一维数组（inputs大概是一个list？）按长度补齐（padding）到相同的长度，并把它们堆叠成一个二维数组返回。"""
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    if maxlen:
        padded = np.stack([pad_data(x, maxlen, PAD) for x in inputs])
    else:
        max_len = max((len(x) for x in inputs))
        padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    """就是把一组二维数组——通常是每条样本的 Mel‐谱图或声学特征——在“时长”这个维度上统一补齐到相同长度，然后一同堆成一个三维批次张量，"""
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_3D(inputs, B, T, L):
    """若干个二维矩阵（比如每条音频的 attention‐prior 矩阵，尺寸各不相同）补齐到相同的大小，然后堆成一个三维的 batch 张量"""
    inputs_padded = np.zeros((B, T, L), dtype=np.float32)
    for i, input_ in enumerate(inputs):
        inputs_padded[i, :np.shape(input_)[0], :np.shape(input_)[1]] = input_
    return inputs_padded

#源代码：output = pad(output, max_len)
def pad(input_ele, mel_max_length=None):
    """输入参数为output, max_len，分别都是list(有batchsize个元素)，其中outputlist中的元素是一个音频的音素序列特征向量，形状为(该音频预测帧长，256)  ，max_len中的元素为每个音频的预测帧长\n
       输出为out_padded，形状为(B,最大帧长，256)，其中最大帧长为一个batch中音频的最大帧长"""
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )  #在行底部填充max_len - batch.size(0)=max-当前帧数 个行，用常数0.0填充，这样就变成了(batch中最大帧长，256)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
