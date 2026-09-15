import re
import os
import json
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p

from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

#单一音频生成，也就是single才会用
def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

#单一音频生成，也就是single才会用
def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(device, model, args, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:-3]),
                spker_embeds=batch[-3],
                emotions=batch[-2],
                history_info=batch[-1],
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
                id=0,
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                args,
            )

#python synthesize.py --source preprocessed_data/DailyTalk/val_*.txt --restore_step RESTORE_STEP --mode batch --dataset DailyTalk
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #--restore_step：必填 整数
    parser.add_argument("--restore_step", type=int, required=True)
    #--mode：必填 字符串，只能是 "batch" 或 "single"，控制是“批量合成整个数据集”还是“单句合成”。 实际使用的时候可以直接写batch，也可以用双引号包围"batch"
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    #--source：可选，批量模式下指定一个元数据文件（如 val_frame.txt），脚本会根据里面的条目依次合成。
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    #--text：可选，单句模式下直接传入你要合成的原始文本。
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    #--speaker_id：多说话人模型下选择哪个说话人；单句模式有效，默认 "p225"
    parser.add_argument(
        "--speaker_id",
        type=str,
        default="p225",
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    #--emotion_id：多情感模型下选择哪种情感；单句模式有效，默认 "happiness"。
    parser.add_argument(
        "--emotion_id",
        type=str,
        default="happiness",
        help="emotion ID for multi-emotion synthesis, for single-sentence mode only",
    )
    #--dataset：必填，告诉脚本要用哪个数据集
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    #下面三个参数 --pitch_control / --energy_control / --duration_control：可选 浮点数，分别用来在合成时整体放大或缩小基频、能量和时长（说话速度），默认都为1.0。
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    #把命令行里以 --xxx value 形式传入的参数解析到 args 对象里，比如 args.mode、args.restore_step、args.text 等，然后脚本后续就可以用这些属性来决定流程（批量还是单句、用哪个 checkpoint、控制因子怎么设、读哪个 source 文件……）
    args = parser.parse_args()

    # Check source texts，根据batch还是single检查源文本
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    os.makedirs(
        os.path.join(train_config["path"]["result_path"], str(args.restore_step)), exist_ok=True)

    # Set Device
    torch.manual_seed(train_config["seed"])  #torch.manual_seed(...) 会固定 CPU 上所有 PyTorch 随机操作的种子；
    if torch.cuda.is_available():
        torch.cuda.manual_seed(train_config["seed"])  #torch.cuda.manual_seed(train_config["seed"])会固定 GPU 上所有 PyTorch 随机操作的种子；
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Device of CompTransTTS:", device)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config, model_config)
        #batchs是一个可迭代对象，其中每个元素都是经过collate_fn生成的元组，每个元组为一个batch的信息
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        assert model_config["history_encoder"]["type"] == 'none', "Single inference is not supported for conversational TTS, currently"
        ids = raw_texts = [args.text[:100]]

        # Speaker Info
        load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
        speakers = np.array([speaker_map[args.speaker_id]]) if model_config["multi_speaker"] else np.array([0]) # single speaker is allocated 0
        spker_embed = np.load(os.path.join(
            preprocess_config["path"]["preprocessed_path"],
            "spker_embed",
            "{}-spker_embed.npy".format(args.speaker_id),
        )) if load_spker_embed else None

        # Emotion Info
        emotions = None
        if model_config["multi_emotion"]:
            with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json")) as f:
                emotion_map = json.load(f)
            emotions = np.array([emotion_map[args.emotion_id]])

        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        else:
            raise NotImplementedError
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embed, emotions)]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(device, model, args, configs, vocoder, batchs, control_values)
