import torch
import torch.nn as nn
from torchpruner.attributions import ShapleyAttributionMetric
import torchpruner_cifar10 as cifar10
from parameter import get_parameter
import os
import numpy as np
import time
from log_module import log
from resnet import BasicBlock

args = get_parameter()

train_loader, val_loader, test_loader = cifar10.get_dataset_and_loaders(args=args)
loss = cifar10.loss

network = torch.load(os.path.join(args.train_save_path, "%s_check_point.pth" % args.network))

network.cuda(args.device_ids[0])

criterion = nn.CrossEntropyLoss()

attr = ShapleyAttributionMetric(network, val_loader, loss, torch.device("cuda:%d" % args.device_ids[0]), sv_samples=5)

# number of conv
num_conv = 0
for module in network.modules():
    if isinstance(module, BasicBlock):
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
    if isinstance(module, BasicBlock):
        for m in module.children():
            if type(m) == nn.Conv2d:
                run_log(m)

np.save(os.path.join(args.sv_save_path, args.shap_value_path),
        np.array(res, dtype=object))
