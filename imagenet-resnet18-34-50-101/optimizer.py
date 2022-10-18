# -*- coding: UTF-8 -*-
import torch


def get_optimizer(network, args):
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, network.parameters()),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[31, 61], gamma=0.1)

    if args.multi_gpu:
        optimizer = torch.nn.DataParallel(optimizer, device_ids=args.device_ids)
        scheduler = torch.nn.DataParallel(scheduler, device_ids=args.device_ids)

    return optimizer, scheduler
