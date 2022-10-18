import torch
import dill
network = torch.load('./trained_models/vgg16_check_point.pth')

torch.save(network, 'trained_models/vgg16_check_point.pth', _use_new_zipfile_serialization=False, pickle_module=dill)