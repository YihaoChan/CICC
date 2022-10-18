# -*- coding: UTF-8 -*-
from thop import profile, clever_format
import torch
from log_module import log
import os
from parameter import get_parameter


def load():
    args = get_parameter()

    network = None

    if args.train_flag:
        network = torch.load(os.path.join(args.train_save_path, "%s_check_point.pth" % args.network))
    elif args.prune_flag:
        network = torch.load(os.path.join(args.prune_save_path, "%s_check_point.pth" % args.network))

    network = network.cuda(args.device_ids[0])

    return args, network


def main():
    args, network = load()

    x = torch.randn(1, 3, 32, 32).cuda(args.device_ids[0])

    flops, params = profile(network, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")

    log.logger.info("FLOPs: " + flops)
    log.logger.info("Params: " + params)


if __name__ == '__main__':
    main()
