# 0 Pipeline

step 1. train：做一次完整结构的训练，保存完整的网络结构和模型权重；

step 2. prune：

1. 加载网络结构；
2. 加载step 1中trained了的模型权重；
3. 对网络结构进行剪枝.

step 3. retrain：

在step 2中加载了step 1中已trained了的模型权重，并重新在数据集上训练一次 => 使用新的剪枝了的结构，对已有权重进行微调.

# 1 计算Shapley Value

对训练权重进行计算：

```
python3 torchpruner_attribute.py --train-flag --train-save-path ./trained_models/ --data-path /data/dataset/data.cifar10
```

# 2 训练 + 推理

```
python3 train_eval.py --train-flag --data-set CIFAR10 --data-path /data/dataset/data.cifar10 --train-save-path ./trained_models/
```

# 3 剪枝 + 再训练 + 推理
```
python3 train_eval.py --prune-flag --train-save-path ./trained_models/ --prune-save-path ./trained_models/pruning_results --data-path /data/dataset/data.cifar10 --prune-channels 32 0 0 0 0 0 0 256 256 256 256 256 256 --retrain-flag --retrain-epoch 40 --retrain-lr 0.001
```

在进行传参时：

--prune-channels参数是指裁剪掉的通道数，如果原来通道数为128，而--prune-channels设为3，那么裁剪后的通道数就是125。

因此，论文中裁剪前后的maps数量为128 128的情形，就是传--prune-channels = 0的情况。

# 4 推理 & 计算、分析

## 4.1 加载训练权重

仅加载训练权重并推理的命令如下：

```
python3 evaluate.py --train-flag --data-set CIFAR10 --data-path /data/dataset/data.cifar10 --train-save-path ./trained_models/
```

计算训练权重的MACs、Params的命令如下：

```
python3 cal_macs_params.py --train-flag --train-save-path ./trained_models/
```

计算训练权重的Feature Map的Entropy的命令如下：

```
python3 cal_feature_map_entropy.py --train-flag --data-set CIFAR10 --data-path /data/dataset/data.cifar10 --train-save-path ./trained_models/
```

## 4.2 加载剪枝权重

仅加载剪枝权重并推理的命令如下：

```
python3 evaluate.py --prune-flag --prune-save-path ./trained_models/pruning_results/ --data-path /data/dataset/data.cifar10
```

计算剪枝权重的MACs、Params的命令如下：

```
python3 cal_macs_params.py --prune-flag --prune-save-path ./trained_models/pruning_results/
```
