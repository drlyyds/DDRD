# Dialogue Deep Representation Disentanglement for Conversational Speech Synthesis

This repository provides the implementation and speech demo for Dialogue Deep Representation Disentanglement (DDRD), a framework for multimodal context modeling in conversational speech synthesis.

## Speech Demo

Listen to the generated samples and representative baselines on the [DDRD speech demo](https://drlyyds.github.io/DDRD/).

## Code

The implementation is provided in [`code/`](code/). The main entry points are:

- `prepare_align.py`: prepare the DailyTalk corpus structure.
- `preprocess.py`: extract features and build the processed dataset.
- `train.py`: train DDRD on one or more CUDA GPUs.
- `synthesize.py`: synthesize evaluation samples from a checkpoint.

### Data preparation

Set `path.corpus_path` in `code/config/DailyTalk/preprocess.yaml` to the local DailyTalk directory, then run:

```bash
cd code
python prepare_align.py --dataset DailyTalk
python preprocess.py --dataset DailyTalk
```

Before training, set an integer random seed in `code/config/DailyTalk/train.yaml` and review the output paths and model settings.

### Training

Training requires an NVIDIA GPU with CUDA support.

```bash
cd code
python train.py --dataset DailyTalk
```

To enable automatic mixed precision, add `--use_amp`. To resume from a saved step, add `--restore_step STEP`.

### Synthesis

For batch synthesis from a prepared metadata file:

```bash
cd code
python synthesize.py --restore_step STEP --mode batch --source PATH_TO_METADATA --dataset DailyTalk
```

Model checkpoints and datasets are not included in this repository.
