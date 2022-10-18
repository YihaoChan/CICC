# CICC: Channel Pruning via the Concentration of Information and Contributions of Channels

## Requirements

```
torch 1.7
torchvision
torchpruner==0.1
thop
pip3 install -e git+https://github.com/marcoancona/TorchPruner.git#egg=torchpruner
```

## Pipeline

The pipeline below is for ResNets on CIFAR10, but it is similar for other architectures on CIFAR10 and ImageNet. So get into other directories and adjust the commands.

### Pretrain dense models.

```python
python3 train_eval.py --train-flag --network resnet20/resnet32/resnet56/resnet110
```

Then the pretrained models are saved in the default directory `./trained_models`.

### Calculate Shapley values of the pretrained models on CIFAR10.

```python
python3 torchpruner_attribute.py --train-flag --train-save-path ./trained_models/ --data-path /data/dataset/data.cifar10 --network resnet20/resnet32/resnet56/resnet110
```

Then the Shapley values of the pretrained models are saved in `trained_shap_values`.

### Prune and finetune to obtain sparse models.

```python
python3 train_eval.py --prune-flag --strategy oneshot/iterative_static/iterative_dynamic
```

The layer-wise pruning rate is **hard coded** in `prune.py`.

### Evaluate the performance.

```python
python3 evaluate.py --train-flag/--prune-flag --network resnet20/resnet32/resnet56/resnet110
```

Assume the dense/sparse models are saved in the default directory, otherwise the path should be specified.

### Calculate MACs and parameters.

```python
python3 cal_macs_params.py --train-flag/--prune-flag
```

### Calculate entropy and rank of the pretrained models.

```python
python3 cal_conv_output_entropy.py/cal_conv_output_rank.py --train-flag --network resnet20/resnet32/resnet56/resnet110
```

### Plot the distribution of entropy/rank for the pretrained models.

```python
python3 plot_rank_entropy.py --type entropy/rank --network resnet20/resnet32/resnet56/resnet110 --interval 2
```

### Plot the heat map for batches of images.

```python
python3 rank_entropy_heat.py --type entropy/rank --network resnet20/resnet32/resnet56/resnet110 --interval 2
```

## Acknowledgements

This paper is not the best, but my best.

Thank:

[Awesome-Pruning](https://github.com/he-y/Awesome-Pruning): A paper library to tap into model pruning.

[PFEC](https://github.com/tyui592/Pruning_filters_for_efficient_convnets): An enlightenment paper and code repository for me.

[Torch-Pruning](https://github.com/VainF/Torch-Pruning): An efficient Python module for pruning models.

[TorchPruner](https://github.com/marcoancona/TorchPruner): A Python module to calculate Shapley values in image classification.

[Guanzhong Tian](https://scholar.google.com/citations?user=0q-7PI4AAAAJ&hl=zh-CN): My supervisor.

[Zhishan Li](https://scholar.google.com/citations?user=zyGbNooAAAAJ&hl=en): My second supervisor!

[Yingqing Yang](https://github.com/yingqing0317): Thank you for pulling me out when I was blue.
