import argparse
import os
#os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import yaml
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.cuda import amp

from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import get_configs_of, get_variance_level, to_device, log, synth_one_sample
from model import CompTransTTSLoss
from dataset import Dataset

from evaluate import evaluate
#这是 PyTorch 中 CuDNN 库的配置。CuDNN 是 NVIDIA 的深度神经网络加速库。设置 benchmark 为 True，会让 CuDNN 自动寻找最适合当前硬件的卷积算法，提升计算效率
torch.backends.cudnn.benchmark = True
#torch.backends.cuda.matmul.allow_tf32 = False   # 关 TF32
#torch.backends.cudnn.allow_tf32= False
#torch.backends.cudnn.benchmark = False    # 不让 cuDNN 动态挑 kernel
#torch.backends.cudnn.deterministic = True     # （可选）确保稳定

#train(0, args-命令行参数, configs-所有配置, batch_size, num_gpus==1)
def train(rank, args, configs, batch_size, num_gpus):
    # 从配置中解包预处理配置、模型配置和训练配置
    preprocess_config, model_config, train_config = configs
    # 如果使用多GPU进行训练
    if num_gpus > 1:
        # 初始化分布式训练进程组
        init_process_group(
            # 分布式后端，例如 'nccl'
            backend=train_config["dist_config"]['dist_backend'],
            # 分布式训练的初始化方法，如 'tcp://localhost:12345'
            init_method=train_config["dist_config"]['dist_url'],
            # 总的进程数
            world_size=train_config["dist_config"]['world_size'] * num_gpus,   #1*num_gpus
            # 当前进程的排名
            rank=rank,
        )
    # 定义当前进程使用的设备
    device = torch.device('cuda:{:d}'.format(rank))    #{:d} 明确表示 “十进制整数”,{} 相当于 “用默认格式”

    # 获取数据集
    # 获取pitch用什么级别标签,level_tag=frame
    level_tag, *_ = get_variance_level(preprocess_config, model_config)

    dataset = Dataset(
        "train_{}.txt".format(level_tag), preprocess_config, model_config, train_config, sort=True, drop_last=True
    )
    data_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    #当你想在 DataLoader 里“先拿出一大块样本（大小 = batch_size * group_size），在这块内部做一次长度排序，然后再拆成 group_size 个真正的训练批次”时，就需要把 group_size 设为大于 1。
    assert batch_size * group_size < len(dataset) #assert就是一个安全检查，要求每组的样本数小于数据集总样本数

    #DataLoader 本质上就是在“如何取数据”与“如何拼批次”之间架起一座桥，自动化地把你的 Dataset 输出转换成一小批一小批能直接送进模型的张量。
    #每次会先从 dataset 中顺序取出 batch_size*4 条样本，放进一个“临时大组”里。
    # 取到一个长度为 batch_size*4 的“临时大组”后，DataLoader 会把这份 data = [dataset[i0](因为用[]坐标取默认会调用__getitem__方法), …, dataset[iN]](其中有 batch_size*4个) 一口气传给 dataset.collate_fn(data)。

    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=False,
        sampler=data_sampler,  #sampler=None
        collate_fn=dataset.collate_fn,
    )
    """loader就是一个可迭代对象，每次返回的都是一个 长度为 4 的列表(由dataset类中的collate_fn返回)"""

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    if num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank]).to(device)
    scaler = amp.GradScaler(enabled=args.use_amp) #如果命令行 --use_amp 参数为true，那么开启混合精度训练，否则什么都不做
    Loss = CompTransTTSLoss(preprocess_config, model_config, train_config).to(device)

    # Load vocoder ，HIFIgan的声码器，是关闭训练特性的()

    vocoder = get_vocoder(model_config, device)
    """vocoder声码器的输入是(B,D,80),输出为(B,L)L是采样点个数"""

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]  #1
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"] #1.0
    total_step = train_config["step"]["total_step"] #900000
    """900000"""
    log_step = train_config["step"]["log_step"]
    """100"""
    save_step = train_config["step"]["save_step"]  #25000
    """25000"""
    synth_step = train_config["step"]["synth_step"]  #1000  每隔1000步进行一次语音合成，用于检查模型的合成效果。
    """1000"""
    val_step = train_config["step"]["val_step"]  #1000 每隔1000步进行一次验证，评估模型在验证集上的性能。
    """1000"""

    if rank == 0:
        print("Number of CompTransTTS Parameters: {}\n".format(get_param_num(model)))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"(可训练的参数量)Trainable  : {trainable:,}")
        # Init logger
        for p in train_config["path"].values():  #train_config["path"].values()就会拿到这三个字符串路径
            os.makedirs(p, exist_ok=True) #创建它的目录以及上层目录
        train_log_path = os.path.join(train_config["path"]["log_path"], "train")
        val_log_path = os.path.join(train_config["path"]["log_path"], "val")
        os.makedirs(train_log_path, exist_ok=True)
        os.makedirs(val_log_path, exist_ok=True)

        #分别给训练和验证阶段各开一个 SummaryWriter，后面你就可以用 train_logger.add_scalar(...)、val_logger.add_image(...) 等接口，把 loss、学习率曲线或者示例音频、对齐图都写到 TensorBoard。
        train_logger = SummaryWriter(train_log_path)
        val_logger = SummaryWriter(val_log_path)

        #这三行就是在用tqdm给整个训练过程画一个“外层”进度条，并且支持「从中断步数」继续显示进度
        outer_bar = tqdm(total=total_step, desc="Training", position=0)   #desc="Training"：在进度条前面显示一个标签 “Training”  # position=0：如果你后来还有别的内层进度条，就把这个放在最上面一行。
        outer_bar.n = args.restore_step  #恢复到断点
        outer_bar.update()

    train = True
    batchnum = 1
    while train:
        if rank == 0:
            inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        if num_gpus > 1:
            data_sampler.set_epoch(epoch)

        #loader就是一个可迭代对象，每次返回的都是一个 长度为 4 的列表(由dataset类中的collate_fn返回)
        for batchs in loader:
            if train == False:
                break
            #batchs是一个长度为4的列表，列表中的每一个元素是一个元组(也就是batch)，(ids,raw_texts,speakers,texts,text_lens,max(text_lens),mels,mel_lens,max(mel_lens),pitches,energies,durations,attn_priors,spker_embeds,emotions,history_info,)这是一个batch的所有音频的信息
            for batch in batchs:

                batch = to_device(batch, device)

                with amp.autocast(args.use_amp):
                    # Forward
                    output = model(*(batch[2:]), step=step,id=batchnum)
                    batchnum += 1
                    #batch[9:11]为 pitches,energies   #output[-3:-1]为p_targets和e_targets(音素级pitch和energy)
                    decoupling_loss=output[-1]
                    batch[9:11], output = output[-3:-1], output[:-3] # Update pitch and energy level
                    # Cal Loss
                    losses = Loss(batch, output, step,decoupling_loss)
                    total_loss = losses[0]  #losses 是一个 tuple，第一项就是各个子损失加权后的“总损失”张量
                    total_loss = total_loss / grad_acc_step #没有做梯度累计，因此没效果

                # Backward
                scaler.scale(total_loss).backward()

                # Clipping gradients to avoid gradient explosion
                if step % grad_acc_step == 0:  #每步都要进行梯度裁剪
                    scaler.unscale_(optimizer._optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh) #进行梯度裁剪，保证所有参数的梯度叠加起来的 L₂ 范数不超过1

                # Update weights
                optimizer.step_and_update_lr(scaler)  #根据当前训练步数算出新的学习率并写入底层的 Adam 优化器，然后用刚才累积好的 .grad 去更新模型的参数。
                scaler.update()
                optimizer.zero_grad()  #把所有参数的 .grad 清零，为下一轮累积梯度做好准备。

                if rank == 0:
                    if step % log_step == 0: #每100步记录一次日志
                        losses_ = [sum(l.values()).item() if isinstance(l, dict) else l.item() for l in losses]
                        message1 = "Step {}/{}, ".format(step, total_step)
                        message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, CTC Loss: {:.4f}, Binarization Loss: {:.4f}, Decoupling Loss: {:.4f}".format(
                            *losses_
                        )

                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(message1 + message2 + "\n")   #把损失写入log.txt

                        if step % log_step == 0:
                            outer_bar.write(message1 + message2)  #相当于print在控制台打印，但是会打印在进度条的上方
                            log(train_logger, step, losses=losses) #画出损失图像

                    if step % synth_step == 0: #每隔 1000 步，对当前的一个训练样本做一次“示例合成”（sampling）
                        #tag通常是用来区分是哪条样本，是basename？
                        # fig是梅尔谱图
                        # fig_attn 是一个 Matplotlib Figure 或 NumPy 图像展示每个 Mel 帧到每个音素的对齐概率
                        # wav_reconstruction是根据得到的真实梅尔谱图然后经过声码器合成的
                        # wav_reconstruction是根据预测的梅尔谱图然后经过声码器合成的
                        fig, fig_attn, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                            batch,
                            output,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )
                        if fig_attn is not None:
                            log(
                                train_logger,
                                img=fig_attn,
                                tag="Training_attn/step_{}_{}".format(step, tag),
                            )
                        log(
                            train_logger,
                            img=fig,
                            tag="Training/step_{}_{}".format(step, tag),
                        )
                        sampling_rate = preprocess_config["preprocessing"]["audio"][
                            "sampling_rate"
                        ]
                        log(
                            train_logger,
                            audio=wav_reconstruction,  #这里的wav_reconstruction是根据得到的真实梅尔谱图然后经过声码器合成的
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_reconstructed".format(step, tag),
                        )
                        log(
                            train_logger,
                            audio=wav_prediction,  #这里的wav_reconstruction是根据预测的梅尔谱图然后经过声码器合成的
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_synthesized".format(step, tag),
                        )

                        # 每隔 25000 步把“模型 + 优化器状态”一并写到磁盘，形成一个可恢复的 checkpoint：
                    if step % save_step == 0:
                        torch.save(
                            {
                                "model": model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                            },
                            os.path.join(
                                train_config["path"]["ckpt_path"],
                                "{}.pth.tar".format(step),
                            ),
                        )
                    if step % val_step == 0:
                        model.eval()
                        message = evaluate(device, model, step, configs, val_logger, vocoder, losses)
                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(message + "\n")
                        outer_bar.write(message)

                        model.train()

                if step == total_step:
                    train = False
                    break
                step += 1
                if rank == 0:
                    outer_bar.update(1)

            if rank == 0:
                inner_bar.update(1)
        epoch += 1
        batchnum=1

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CPU training is not allowed."     #检查当前环境是否有可用的 GPU（CUDA），如果没有发现 GPU，就会抛出断言错误并打印 "CPU training is not allowed."，强制要求必须用 GPU 训练。
    parser = argparse.ArgumentParser()
    """向 parser 注册三个名为 --dataset ,--use_amp,--restore_step  的命令行选项。"""

    parser.add_argument('--use_amp', action='store_true')  # 是否启用 自动混合精度（AMP）训练,  如果在命令行里出现了 --use_amp，那么 args.use_amp 就被设为 True；如果没有出现，args.use_amp 默认为 False。
    parser.add_argument("--restore_step", type=int, default=0)   #--restore_step：指定从哪个训练迭代（step）恢复，默认是 0（不恢复，重新从头训练）。
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()    #args = parser.parse_args()：真正解析命令行，这样可以调用args.dataset，args.use_amp ，args.restore_step

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)  #args.dataset="dailytalk"
    configs = (preprocess_config, model_config, train_config)

    # Set Device
    #统一设置 PyTorch 在 CPU 和 GPU 上的随机数种子，保证训练过程可复现。
    torch.manual_seed(train_config["seed"])
    torch.cuda.manual_seed(train_config["seed"])
    num_gpus = torch.cuda.device_count()   #检测当前机器上有多少块 GPU。
    batch_size = int(train_config["optimizer"]["batch_size"] / num_gpus)  #如果要做多 GPU 训练，通常会把全局 Batch Size 平均分配到每块卡上,这样 batch_size 就是 每张卡 上实际跑的样本数。

    torch.autograd.set_detect_anomaly(True)

    # Log Configuration
    print("\n==================================== Training Configuration ====================================")
    print(' ---> Automatic Mixed Precision:', args.use_amp)
    print(' ---> Number of used GPU:', num_gpus)
    print(' ---> Batch size per GPU:', batch_size)
    print(' ---> Batch size in total:', batch_size * num_gpus)
    print(" ---> Type of Building Block:", model_config["block_type"])
    print(" ---> Type of Duration Modeling:", "unsupervised" if model_config["duration_modeling"]["learn_alignment"] else "supervised")
    print("=================================================================================================")
    print("Prepare training ...")

    #如果检测到有多于 1 块 GPU，就进入分布式模式。
    if num_gpus > 1:
        mp.spawn(train , nprocs=num_gpus, args=(args, configs, batch_size, num_gpus))
        """ train               # 要并行执行的函数
         nprocs=num_gpus,       # 子进程数量 = GPU 数量
        args=(                 # 传给 train 的额外参数
            args,              # 命令行参数
            configs,           # 三份 config
            batch_size,        # 每卡 batch size
            num_gpus           # 总卡数
        )      """
    else:
        train(0, args, configs, batch_size, num_gpus)
