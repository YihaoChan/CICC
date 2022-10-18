import torch
import torch.nn as nn
from torchpruner.attributions import ShapleyAttributionMetric
from parameter import get_parameter
from vgg import VGG
import torchpruner_cifar10 as cifar10
import os
import numpy as np
import time
from log_module import log

args = get_parameter()

train_loader, val_loader, test_loader = cifar10.get_dataset_and_loaders(args=args)
loss = cifar10.loss

VGG.forward = cifar10.vgg_forward_partial
VGG.forward_partial = cifar10.vgg_forward_partial

network = torch.load(os.path.join(args.train_save_path, "%s_check_point.pth" % args.network))

network.cuda(args.device_ids[0])

criterion = nn.CrossEntropyLoss()

layers = list(network.features.children())

attr = ShapleyAttributionMetric(network, val_loader, loss, torch.device("cuda:%d" % args.device_ids[0]), sv_samples=5)

# number of conv
num_conv = 0
for module in layers:
    if type(module) == nn.Conv2d:
        num_conv += 1

# calculate shapley value for Conv modules
res = []
conv_progress = 0
for module in layers:
    if type(module) == nn.Conv2d:
        logs_ = '%s: ' % time.ctime()
        logs_ += 'Module %s, ' % (str(module))
        conv_progress += 1
        logs_ += 'Iteration [%d/%d]' % (conv_progress, num_conv)
        log.logger.info(logs_)

        scores = attr.run(module)
        res.append(scores)

np.save(os.path.join(args.sv_save_path, args.shap_value_path),
        np.array(res, dtype=object))
