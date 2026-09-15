import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]  #"D:\\PythonProject\\DailyTalk-main\\"
    sub_dir = config["path"]["sub_dir_name"]  #"data"
    out_dir = config["path"]["raw_path"]   #"./raw_data/DailyTalk"
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]   #22050
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]   #32768.0
    cleaners = config["preprocessing"]["text"]["text_cleaners"]  #["english_cleaners"]

    for turn_name in tqdm(os.listdir(os.path.join(in_dir, sub_dir))):
        for file_name in os.listdir(os.path.join(in_dir, sub_dir, turn_name)):
            if "wav" not in file_name:
                continue
            base_name = file_name.replace(".wav", "")
            text_path = os.path.join(
                in_dir, sub_dir, turn_name, "{}.txt".format(base_name)
            )
            wav_path = os.path.join(
                in_dir, sub_dir, turn_name, "{}.wav".format(base_name)
            )
            with open(text_path,encoding="utf-8") as f:
                text = f.readline().strip("\n")
            text = _clean_text(text, cleaners)

            os.makedirs(os.path.join(out_dir, sub_dir, turn_name), exist_ok=True)
            wav, _ = librosa.load(wav_path,sr=sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, sub_dir, turn_name, "{}.wav".format(base_name)),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(
                os.path.join(out_dir, sub_dir, turn_name, "{}.lab".format(base_name)),
                "w",
            ) as f1:
                f1.write(text)
