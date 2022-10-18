# -*- coding: UTF-8 -*-
import time
import torch
from utils import AverageMeter, get_data_loader
from log_module import log
from parameter import get_parameter
import os


def test_network(args, network=None):
    if network is None:  # 只推理，没有上游传下来的模型
        if args.train_flag:
            network = torch.load(os.path.join(args.train_save_path, "densenet40_check_point.pth"),
                                 map_location='cuda:%s' % args.device_ids[0])
        if args.prune_flag:
            network = torch.load(os.path.join(args.prune_save_path, "densenet40_check_point.pth"),
                                 map_location='cuda:%s' % args.device_ids[0])

    # train做了分布式的网络传过来，所以不用再做分布式了
    network = network.cuda(args.device_ids[0])

    _, test_data_loader = get_data_loader(args)

    test_step(args, network, test_data_loader)


def test_step(args, network, test_data_loader):
    network.eval()

    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for inputs, targets in test_data_loader:
            inputs, targets = inputs.cuda(args.device_ids[0]), targets.cuda(args.device_ids[0])

            outputs = network(inputs)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))

            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

    str_ = '%s: Test information, ' % time.ctime()
    str_ += 'Top1: %2.3f, Top5: %2.3f' % (top1.avg, top5.avg)
    log.logger.info("=" * 10 + "Evaluate network densenet40" + "=" * 10)
    log.logger.info(str_)


def accuracy(output, target, topk=(1,)):
    """
    Top-1，Top-5中的Top指的是一个图片中的概率前1和前5，不是所有图片中预测最好的1个或5个图片
    比如一共需要分10类，每次分类器的输出结果都是10个相加为1的概率值，
    Top1就是这十个值中最大的那个概率值对应的分类恰好正确的频率，
    而Top5则是在十个概率值中从大到小排序出前五个，然后看看这前五个分类中是否存在那个正确分类，再计算频率。
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:  # k: 1, 5
        # 若[0.6, 0.2, 0.1, 0.05, 0.05]，那么top1 = 0.6，top3 = 0.9。因此，top1表示判断属于某一类的最大概率值
        # 而换做是correct，则top1表示预测正确的最大概率值
        correct_k = correct[:k].reshape(-1).float().sum(0)  # 前k处的概率之和
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def main():
    args = get_parameter()

    test_network(args, network=None)


if __name__ == '__main__':
    main()
