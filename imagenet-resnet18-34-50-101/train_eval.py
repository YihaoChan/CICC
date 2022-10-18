# -*- coding: UTF-8 -*-
from parameter import get_parameter
from train import train_network
from evaluate import test_network
from prune import prune_network
import torch
import os

if __name__ == '__main__':
    args = get_parameter()

    network = None

    if args.train_flag:
        network = train_network(args, network=None)

    if args.resume_prune_flag:
        network = torch.load(os.path.join(args.prune_save_path, "%s_check_point.pth" % args.network),
                             map_location='cuda:%s' % args.device_ids[0])
        network = train_network(args, network=network)

    if args.prune_flag:
        network = prune_network(args, network=None)

    test_network(args, network=network)
