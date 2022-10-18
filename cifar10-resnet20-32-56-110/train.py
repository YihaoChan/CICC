# -*- coding: UTF-8 -*-
import time
import torch
from loss import LossCalculator
from evaluate import accuracy
from utils import AverageMeter, get_data_loader, check_network
from optimizer import get_optimizer
from log_module import log
import os
import dill


def train_network(args, network=None):
    save_path = ''

    if args.train_flag:
        assert network is None  # network为None，说明是第一次传参，即为train阶段
        save_path = args.train_save_path

        network = check_network(args)

        log.logger.info("=" * 10 + "Train network %s" % args.network + "=" * 10)
    elif args.prune_flag:
        assert network is not None
        save_path = args.prune_save_path

        log.logger.info("=" * 10 + "Retrain network %s" % args.network + "=" * 10)

    if args.multi_gpu:
        network = torch.nn.DataParallel(network, device_ids=args.device_ids)  # 数据并行
    network = network.cuda(args.device_ids[0])  # 模型无法并行，加载到0号卡

    train_data_loader, _ = get_data_loader(args)
    loss_calculator = LossCalculator()
    optimizer, scheduler = get_optimizer(network, args)

    for epoch in range(args.start_epoch, args.epoch):
        # train one epoch
        train_step(args, network, train_data_loader, loss_calculator, optimizer, scheduler, epoch, args.print_freq)

        # adjust learning rate
        if args.multi_gpu:
            scheduler.module.step()
        else:
            scheduler.step()

        model_save_path = os.path.join(save_path, "%s_check_point.pth" % args.network)
        if args.multi_gpu:
            torch.save(network.module, model_save_path, pickle_module=dill)
        else:
            torch.save(network, model_save_path, pickle_module=dill)

    return network


def train_step(args, network, train_data_loader, loss_calculator, optimizer, scheduler, epoch, print_freq=100):
    network.train()

    # set benchmark flag to faster runtime
    torch.backends.cudnn.benchmark = True

    top1 = AverageMeter()
    top5 = AverageMeter()

    for iteration, (inputs, targets) in enumerate(train_data_loader):
        inputs, targets = inputs.cuda(args.device_ids[0]), targets.cuda(args.device_ids[0])

        outputs = network(inputs)

        loss = loss_calculator.calc_loss(outputs, targets)

        if args.multi_gpu:
            optimizer.module.zero_grad()  # 分布式要有module
        else:
            optimizer.zero_grad()

        loss.backward()

        if args.multi_gpu:
            optimizer.module.step()  # 分布式要有module
        else:
            optimizer.step()

        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if iteration % print_freq == 0:
            logs_ = '%s: ' % time.ctime()
            logs_ += 'Epoch [%d], ' % epoch
            if args.multi_gpu:
                logs_ += 'LR [%2.3f], ' % float(scheduler.module.get_lr()[0])
            else:
                logs_ += 'LR [%2.3f], ' % float(scheduler.get_lr()[0])
            logs_ += 'Iteration [%d/%d], ' % (iteration, len(train_data_loader))
            logs_ += 'Top1: %2.3f, Top5: %2.4f, ' % (top1.avg, top5.avg)
            logs_ += 'Loss: %2.3f' % loss_calculator.get_loss_log()
            log.logger.info(logs_)
