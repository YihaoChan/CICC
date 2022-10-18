# -*- coding: UTF-8 -*-
import torch
import torchvision.models.resnet as resnet
import torch_pruning as tp
from log_module import log
from train import train_network
import numpy as np
import os
from channels_selection import *


def prune_network(args, network=None):
    if network is None:
        # 加载训练阶段完整模型权重
        network = torch.load(os.path.join(args.train_save_path, "%s_check_point.pth" % args.network))

    log.logger.info("=" * 10 + "Prune network by shap value" + "=" * 10)

    # 拿已有的已经训练好的模型对结构进行剪枝，输入需要剪的层号和通道数量
    network = prune_step(args, network)

    # 此处不分布式，因为等会传到train里面会做分布式，如果这里做了就重复了
    network = network.cuda(args.device_ids[0])

    # update arguments for retraining pruned network
    args.epoch = args.retrain_epoch
    args.lr = args.retrain_lr

    network = train_network(args, network)  # prune后的网络，再训练一次，微调

    return network


def prune_step(args, network):
    network = network.cuda(args.device_ids[0])

    prune_channels = []
    if args.network == 'resnet18':
        prune_channels = [0] * 16
        assign_prune_channels_resnet18(prune_channels,
                                       layer1_prune=8,
                                       layer2_prune=24,
                                       layer3_prune=48,
                                       layer4_prune=80)
    elif args.network == 'resnet34':
        prune_channels = [0] * 32
        assign_prune_channels_resnet34(prune_channels,
                                       layer1_prune=8,
                                       layer2_prune=12,
                                       layer3_prune=24,
                                       layer4_prune=32)
    elif args.network == 'resnet50':
        prune_channels = [0] * 48
        assign_prune_channels_resnet50(prune_channels,
                                       layer1_prune=8,
                                       layer2_prune=48,
                                       layer3_prune=64,
                                       layer4_prune=80)
    elif args.network == 'resnet101':
        prune_channels = [0] * 99
        assign_prune_channels_resnet101(prune_channels,
                                        layer1_prune=16,
                                        layer2_prune=32,
                                        layer3_prune=42,
                                        layer4_prune=128)
    else:
        raise RuntimeError("Haven't supported this model!")

    shap_value = np.load(os.path.join(args.sv_save_path, args.shap_value_path), allow_pickle=True)

    prune_count = 0

    if args.network == 'resnet18' or args.network == 'resnet34':
        block_type = resnet.BasicBlock
    elif args.network == 'resnet50' or args.network == 'resnet101':
        block_type = resnet.Bottleneck
    else:
        raise RuntimeError("Haven't support such network!")

    DG = tp.DependencyGraph()
    DG.build_dependency(network, example_inputs=torch.randn(1, 3, 32, 32))

    for module in network.modules():
        if isinstance(module, block_type):
            channel_index = get_channel_index(prune_channels[prune_count], shap_value[prune_count])
            pruning_plan = DG.get_pruning_plan(module.conv1, tp.prune_conv, channel_index)
            pruning_plan.exec()
            prune_count += 1

            channel_index = get_channel_index(prune_channels[prune_count], shap_value[prune_count])
            pruning_plan = DG.get_pruning_plan(module.conv2, tp.prune_conv, channel_index)
            pruning_plan.exec()
            prune_count += 1

            if block_type == resnet.Bottleneck:
                channel_index = get_channel_index(prune_channels[prune_count], shap_value[prune_count])
                pruning_plan = DG.get_pruning_plan(module.conv3, tp.prune_conv, channel_index)
                pruning_plan.exec()
                prune_count += 1

    return network


def get_channel_index(num_elimination, sv):
    """
    num_elimination: 剪掉的数量
    排序，返回截取后的索引
    """
    sv = torch.from_numpy(sv)

    # ori: [4, 2, 3, 1] => values=tensor([1, 2, 3, 4]), indices=tensor([3, 1, 2, 0]))
    # 绝对值排序，按照最小值挑出前num_elimination个的下标
    vals, indices = torch.sort(sv)

    return indices[:num_elimination].tolist()
