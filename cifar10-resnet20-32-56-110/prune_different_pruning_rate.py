# -*- coding: UTF-8 -*-
import torch
import torch_pruning as tp
from log_module import log
from train import train_network
import os
from resnet import BasicBlock
import numpy as np
import dill


def prune_network(args, network=None):
    if network is None:
        # 加载训练阶段完整模型权重
        network = torch.load(os.path.join(args.train_save_path, "%s_check_point.pth" % args.network))

    log.logger.info(
        "=" * 10 + "Ablation for pruning rate (%s) of %s" % (str(args.global_pruning_rate), args.network) + "=" * 10)

    # 拿已有的已经训练好的模型对结构进行剪枝
    network = oneshot(args, network)

    # 此处不分布式，因为等会传到train里面会做分布式，如果这里做了就重复了
    network = network.cuda(args.device_ids[0])

    # update arguments for retraining pruned network
    args.epoch = args.retrain_epoch
    args.lr = args.retrain_lr

    network = train_network(args, network)  # prune后的网络，再训练一次，微调

    return network


def oneshot(args, network):
    network = network.cuda(args.device_ids[0])

    pruning_rate = args.global_pruning_rate

    shap_value = np.load(os.path.join(args.sv_save_path, args.shap_value_path), allow_pickle=True)

    prune_count = 0

    DG = tp.DependencyGraph()
    DG.build_dependency(network, example_inputs=torch.randn(1, 3, 32, 32))

    for module in network.modules():
        if isinstance(module, BasicBlock):
            channel_index = get_channel_index(int(pruning_rate * module.conv1.out_channels), shap_value[prune_count])
            pruning_plan = DG.get_pruning_plan(module.conv1, tp.prune_conv, idxs=channel_index)
            pruning_plan.exec()
            prune_count += 2

    return network


def get_channel_index(num_elimination, sv):
    """
    kernel: network.features[i]里的Conv模块
    num_elimination: prune-channels数组: [1 1] => 保留排序后的前几个，此处保留第一个
    get cadidate channel index for pruning，按weight排序，返回截取后的索引
    获取用于修剪的候选通道索引
    """
    sv = torch.from_numpy(sv)

    # ori: [4, 2, 3, 1] => values=tensor([1, 2, 3, 4]), indices=tensor([3, 1, 2, 0]))
    # 绝对值排序，按照最小值挑出前num_elimination个的下标
    vals, indices = torch.sort(sv)

    return indices[:num_elimination].tolist()
