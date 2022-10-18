# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch_pruning as tp
from log_module import log
from train import train_network
import numpy as np
import os
from channels_selection import assign_prune_channels
import time
from torchpruner.attributions import ShapleyAttributionMetric
import torchpruner_cifar10 as cifar10


def prune_network(args, network=None):
    if network is None:
        # 加载训练阶段完整模型权重
        network = torch.load(os.path.join(args.train_save_path, "%s_check_point.pth" % args.network))

    if args.strategy == 'oneshot':
        prune_step = oneshot
    elif args.strategy == 'iterative_static':
        prune_step = iterative_static
    elif args.strategy == 'iterative_dynamic':
        prune_step = iterative_dynamic
    else:
        raise RuntimeError("Incorrect strategy!")

    log.logger.info("=" * 10 + "Prune network %s by %s" % (args.network, args.strategy) + "=" * 10)

    # 拿已有的已经训练好的模型对结构进行剪枝
    network = prune_step(args, network)

    # 此处不分布式，因为等会传到train里面会做分布式，如果这里做了就重复了
    network = network.cuda(args.device_ids[0])

    # update arguments for retraing pruned network
    args.epoch = args.retrain_epoch
    args.lr = args.retrain_lr

    network = train_network(args, network)  # prune后的网络，再训练一次，微调

    return network


def oneshot(args, network):
    network = network.cuda(args.device_ids[0])

    prune_channels = [0] * 13
    assign_prune_channels(prune_channels,
                          entity0_prune=8,
                          entity1_prune=32,
                          entity2_prune=80,
                          entity3_prune=256)

    shap_value = np.load(os.path.join(args.sv_save_path, args.shap_value_path), allow_pickle=True)

    prune_count = 0

    DG = tp.DependencyGraph()
    DG.build_dependency(network, example_inputs=torch.randn(1, 3, 32, 32))

    for module in network.modules():
        if isinstance(module, nn.Conv2d):
            channel_index = get_channel_index(prune_channels[prune_count], shap_value[prune_count])

            pruning_plan = DG.get_pruning_plan(module, tp.prune_conv, idxs=channel_index)
            pruning_plan.exec()

            prune_count += 1

    return network


def iterative_static(args, network):
    network = network.cuda(args.device_ids[0])

    prune_channels = [0] * 13
    assign_prune_channels(prune_channels,
                          entity0_prune=8,
                          entity1_prune=32,
                          entity2_prune=80,
                          entity3_prune=256)

    shap_value = np.load(os.path.join(args.sv_save_path, args.shap_value_path), allow_pickle=True)
    assert len(prune_channels) == len(shap_value)

    prune_count = 0

    DG = tp.DependencyGraph()
    DG.build_dependency(network, example_inputs=torch.randn(1, 3, 32, 32))

    iter_count = 0
    sum_iter = 13 * args.finetune_epoch

    args.epoch = args.finetune_epoch  # 逐层微调的轮数

    for module in network.modules():
        if isinstance(module, nn.Conv2d):
            channel_index = get_channel_index(prune_channels[prune_count], shap_value[prune_count])

            pruning_plan = DG.get_pruning_plan(module, tp.prune_conv, idxs=channel_index)
            pruning_plan.exec()

            prune_count += 1

            logs_ = '%s: ' % time.ctime()
            iter_count += args.finetune_epoch
            logs_ += 'Finetune epoch [%d/%d]' % (iter_count, sum_iter)
            log.logger.info(logs_)

            # finetune
            network = train_network(args, network)

            if args.multi_gpu:
                network = network.module  # reset

    return network


def iterative_dynamic(args, network):
    network = network.cuda(args.device_ids[0])

    prune_channels = [0] * 13
    assign_prune_channels(prune_channels,
                          entity0_prune=16,
                          entity1_prune=96,
                          entity2_prune=112,
                          entity3_prune=64)

    train_loader, val_loader, test_loader = cifar10.get_dataset_and_loaders(args=args)
    loss = cifar10.loss
    attr = ShapleyAttributionMetric(network,
                                    val_loader,
                                    loss,
                                    torch.device("cuda:%d" % args.device_ids[0]),
                                    sv_samples=5)

    prune_count = 0

    DG = tp.DependencyGraph()

    iter_count = 0
    sum_iter = 13 * args.finetune_epoch

    args.epoch = args.finetune_epoch  # 逐层微调的轮数

    for module in network.modules():
        if isinstance(module, nn.Conv2d):
            shap_value = attr.run(module)

            channel_index = get_channel_index(prune_channels[prune_count], shap_value)

            DG.build_dependency(network, example_inputs=torch.randn(1, 3, 32, 32))
            pruning_plan = DG.get_pruning_plan(module, tp.prune_conv, idxs=channel_index)
            pruning_plan.exec()

            prune_count += 1

            logs_ = '%s: ' % time.ctime()
            iter_count += args.finetune_epoch
            logs_ += 'Finetune epoch [%d/%d]' % (iter_count, sum_iter)
            log.logger.info(logs_)

            # finetune
            network = train_network(args, network)

            if args.multi_gpu:
                network = network.module  # reset

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
