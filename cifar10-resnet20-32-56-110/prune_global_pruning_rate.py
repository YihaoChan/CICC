# -*- coding: UTF-8 -*-
import torch
import torch_pruning as tp
from log_module import log
from train import train_network
import os
from resnet import BasicBlock
import numpy as np
import time
from torchpruner.attributions import ShapleyAttributionMetric
import torchpruner_cifar10 as cifar10
import dill


def prune_network(args, network=None):
    assert args.network == 'resnet56'

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

    log.logger.info("=" * 10 + "Ablation for same pruning rate of ResNet-56" + "=" * 10)

    # 拿已有的已经训练好的模型对结构进行剪枝
    network = prune_step(args, network)

    # 此处不分布式，因为等会传到train里面会做分布式，如果这里做了就重复了
    network = network.cuda(args.device_ids[0])

    # update arguments for retraining pruned network
    args.epoch = args.retrain_epoch
    args.lr = args.retrain_lr

    network = train_network(args, network)  # prune后的网络，再训练一次，微调

    return network


def oneshot(args, network):
    network = network.cuda(args.device_ids[0])

    pruning_rate = 0.48

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


def iterative_static(args, network):
    network = network.cuda(args.device_ids[0])

    pruning_rate = 0.48
    layer_num = 54

    shap_value = np.load(os.path.join(args.sv_save_path, args.shap_value_path), allow_pickle=True)

    prune_count = 0

    DG = tp.DependencyGraph()
    DG.build_dependency(network, example_inputs=torch.randn(1, 3, 32, 32))

    iter_count = 0
    sum_iter = layer_num * args.finetune_epoch // 2  # only prune conv1, skip conv2

    args.epoch = args.finetune_epoch
    for module in network.modules():
        if isinstance(module, BasicBlock):
            channel_index = get_channel_index(int(pruning_rate * module.conv1.out_channels), shap_value[prune_count])
            pruning_plan = DG.get_pruning_plan(module.conv1, tp.prune_conv, idxs=channel_index)
            pruning_plan.exec()
            prune_count += 2  # only prune conv1, skip conv2

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

    pruning_rate = 0.48
    layer_num = 54

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
    sum_iter = layer_num * args.finetune_epoch // 2  # only prune conv1, skip conv2

    args.epoch = args.finetune_epoch
    for module in network.modules():
        if isinstance(module, BasicBlock):
            op_conv = module.conv1

            shap_value = attr.run(op_conv)

            channel_index = get_channel_index(int(pruning_rate * module.conv1.out_channels), shap_value)

            DG.build_dependency(network, example_inputs=torch.randn(1, 3, 32, 32))
            pruning_plan = DG.get_pruning_plan(op_conv, tp.prune_conv, idxs=channel_index)
            pruning_plan.exec()

            prune_count += 2  # only prune conv1, skip conv2

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
