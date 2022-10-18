# -*- coding: UTF-8 -*-
import torch
from utils import get_data_loader
from parameter import get_parameter
import os
from tqdm import tqdm
import torchvision.models.resnet as resnet
import torch.nn as nn
from log_module import log
import numpy as np


def load():
    args = get_parameter()

    network = torch.load(os.path.join(args.train_save_path, "%s_check_point.pth" % args.network))

    network = network.cuda(args.device_ids[0])

    _, test_data_loader = get_data_loader(args)

    return args, network, test_data_loader


def test(args, network, test_data_loader, batch_limit):
    network.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_data_loader):
            inputs, targets = inputs.cuda(args.device_ids[0]), targets.cuda(args.device_ids[0])
            network(inputs)  # forward propagation to activate the hooks to append the rank of feature maps
            if (batch_idx + 1) == batch_limit:
                break


def calculate(args, network, test_data_loader):
    conv_outputs = []

    def hook_feature_map_rank(module, inputs, output):
        """
        hook function
        """
        conv_outputs.append(output)

    conv_module_num = 0  # count the number of convolution modules

    if args.network == 'resnet18' or args.network == 'resnet34':
        block_type = resnet.BasicBlock
    elif args.network == 'resnet50' or args.network == 'resnet101':
        block_type = resnet.Bottleneck
    else:
        raise RuntimeError("Haven't support such network!")

    for module in network.modules():
        if isinstance(module, block_type):
            for m in module.children():
                # Block: Conv -> BN -> [ReLU]
                if type(m) == nn.BatchNorm2d:
                    conv_module_num += 1
                    m.register_forward_hook(hook_feature_map_rank)  # inject the hook

    batch_limit = [i + 1 for i in range(8)]
    for limit in batch_limit:
        # inference to make the hooks work
        test(args, network, test_data_loader, limit)
        mean_conv_outputs = [[] for _ in range(conv_module_num)]

        assert len(conv_outputs) % conv_module_num == 0
        batches = len(conv_outputs) // conv_module_num  # number of batches that are collected

        for idx in range(conv_module_num):
            for turn in range(batches):
                mean_conv_outputs[idx].append(conv_outputs[idx + turn * conv_module_num])

        for idx in range(conv_module_num):
            mean_conv_outputs[idx] = torch.mean(torch.stack(mean_conv_outputs[idx], dim=0), dim=0)

        cal_rank(args, mean_conv_outputs, limit)
        conv_outputs.clear()


def cal_rank(args, items, batch_num):
    res = []

    for tensor in tqdm(items):
        # Only with ReLU can the ranks not be full, cuz ReLU makes the negative weights be zero.
        tensor = nn.ReLU(inplace=True)(tensor)  # mimic self.ReLU, instead of F.relu()

        batch_size = tensor.shape[0]
        out_channels = tensor.shape[1]

        # reference: https://github.com/lmbxmu/HRank/blob/master/rank_generation.py
        rank = torch.tensor(
            [torch.matrix_rank(tensor[i, j, :, :]).item()
             for i in range(batch_size) for j in range(out_channels)]
        )

        # [batch_size, out_channels]
        rank = rank.reshape(batch_size, -1).float()

        # [out_channels]
        rank = rank.sum(0)

        # 整个conv层output的秩
        conv_output_rank = torch.sum(rank)

        # batch_size是固定的，但每个conv层的通道数都不同，所以要除以通道数，得出平均每个通道的秩
        mean_channel_rank = conv_output_rank / out_channels

        res.append(mean_channel_rank)

    save_dir = os.path.join('trained_rank_npy/%s/batches_%d' % (args.network, batch_num))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, '%s_rank.npy' % args.network), res)

    log.logger.info("=" * 10 + "Rank of full-trained %s under batch %d" % (args.network, batch_num) + "=" * 10)
    log.logger.info(" ".join(str(round(float(i), 4)) for i in res))


def main():
    args, network, data_loader = load()

    calculate(args, network, data_loader)


if __name__ == '__main__':
    main()
