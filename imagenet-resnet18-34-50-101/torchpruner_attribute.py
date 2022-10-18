import torch
import torch.nn as nn
from torchpruner.attributions import ShapleyAttributionMetric
from parameter import get_parameter
import torchpruner_imagenet as imagenet
import os
import numpy as np
import time
from log_module import log
import torchvision.models.resnet as resnet

args = get_parameter()

train_loader, val_loader, test_loader = imagenet.get_dataset_and_loaders(args=args)
loss = imagenet.loss

network = torch.load(os.path.join(args.train_save_path, "%s_check_point.pth" % args.network))
network.cuda(args.device_ids[0])

attr = ShapleyAttributionMetric(network, val_loader, loss, torch.device("cuda:%d" % args.device_ids[0]), sv_samples=5)

if args.network == 'resnet18' or args.network == 'resnet34':
    block_type = resnet.BasicBlock
elif args.network == 'resnet50' or args.network == 'resnet101':
    block_type = resnet.Bottleneck
else:
    raise RuntimeError("Haven't support such network!")

# number of conv
num_conv = 0
for module in network.modules():
    if isinstance(module, block_type):
        for m in module.children():
            if type(m) == nn.Conv2d:
                num_conv += 1

res = []
conv_progress = 0


def run_log(mdl):
    global conv_progress

    logs_ = '%s: ' % time.ctime()
    logs_ += 'Module %s, ' % (str(mdl))
    conv_progress += 1
    logs_ += 'Iteration [%d/%d]' % (conv_progress, num_conv)
    log.logger.info(logs_)

    scores = attr.run(mdl)
    res.append(scores)


# calculate shapley value for Conv modules
for module in network.modules():
    if isinstance(module, block_type):
        for m in module.children():
            if type(m) == nn.Conv2d:
                run_log(m)

np.save(os.path.join(args.sv_save_path, args.shap_value_path),
        np.array(res, dtype=object))
