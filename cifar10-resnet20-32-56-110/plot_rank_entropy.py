from matplotlib import pyplot as plt
import numpy as np
import os
import argparse
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import ScalarFormatter
from channels_selection import layer_wise_assign_prune_channels
from sklearn import preprocessing


def load():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='D:\\programme\\BMVC\\images\\rank_entropy\\stage')
    parser.add_argument('--network', type=str, default='resnet20')
    parser.add_argument('--type', type=str, default='rank')
    parser.add_argument('--interval', type=int, default=2)

    args = parser.parse_args()

    return args


def plot_rank_entropy(args):
    feature_maps_dir = 'trained_%s_npy\\%s\\batches_4' % (args.type, args.network)
    feature_maps_path = os.path.join(feature_maps_dir, '%s_%s.npy' % (args.network, args.type))
    feature_maps_item = np.array(np.load(feature_maps_path)).reshape(-1, 1)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(1, 10), copy=True)
    scaled_item = min_max_scaler.fit_transform(feature_maps_item).reshape(-1).tolist()

    prune_channels = []
    flags = [0, 1, 2]
    if args.network == 'resnet20':
        prune_channels = [0] * 18
        layer_wise_assign_prune_channels(prune_channels,
                                         layer1_prune=flags[0],
                                         layer2_prune=flags[1],
                                         layer3_prune=flags[2],
                                         num_block=3)
    elif args.network == 'resnet32':
        prune_channels = [0] * 30
        layer_wise_assign_prune_channels(prune_channels,
                                         layer1_prune=flags[0],
                                         layer2_prune=flags[1],
                                         layer3_prune=flags[2],
                                         num_block=5)
    elif args.network == 'resnet56':
        prune_channels = [0] * 54
        layer_wise_assign_prune_channels(prune_channels,
                                         layer1_prune=flags[0],
                                         layer2_prune=flags[1],
                                         layer3_prune=flags[2],
                                         num_block=9)
    elif args.network == 'resnet110':
        prune_channels = [0] * 108
        layer_wise_assign_prune_channels(prune_channels,
                                         layer1_prune=flags[0],
                                         layer2_prune=flags[1],
                                         layer3_prune=flags[2],
                                         num_block=18)

    x1 = [idx + 1 for idx, flag in enumerate(prune_channels) if flag == flags[0]]
    x2 = [idx + 1 for idx, flag in enumerate(prune_channels) if flag == flags[1]]
    x3 = [idx + 1 for idx, flag in enumerate(prune_channels) if flag == flags[2]]

    y1, y2, y3 = [], [], []

    for idx, item in enumerate(scaled_item):
        if prune_channels[idx] == flags[0]:
            y1.append(item)
        elif prune_channels[idx] == flags[1]:
            y2.append(item)
        elif prune_channels[idx] == flags[2]:
            y3.append(item)

    # plot
    plt.style.use('seaborn-ticks')
    plt.rc('font', family='Times New Roman')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig_path = os.path.join(args.path, '%s_%s.eps' % (args.network, args.type))

    ret1 = plt.bar(x1, y1, color="#426EB4", linewidth=1.0)
    ret2 = plt.bar(x2, y2, color="#3A2885", linewidth=1.0)
    ret3 = plt.bar(x3, y3, color="#00A4AC", linewidth=1.0)
    plt.legend((ret1, ret2, ret3), ('Stage-1', 'Stage-2', 'Stage-3'), loc='upper right', prop={'size': 22})

    class ScalarFormatterForceFormat(ScalarFormatter):
        def _set_format(self):  # Override function that finds format to use.
            self.format = "%1.0f"  # Give format here

    plt.xticks(np.arange(1, len(feature_maps_item) + 1, args.interval),
               np.arange(1, len(feature_maps_item) + 1, args.interval))

    yfmt = ScalarFormatterForceFormat()
    yfmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(yfmt)
    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')

    plt.xlim(0, None)

    plt.tick_params(labelsize=18)
    ax.yaxis.get_offset_text().set_fontsize(18)

    # 设置x、y轴标签
    ax.set_xlabel("Layer", fontsize=26)
    ax.set_ylabel("Scaled %s" % args.type.capitalize(), fontsize=26)

    plt.savefig(save_fig_path, format='eps', bbox_inches='tight', pad_inches=0)

    plt.close()


def plot_fusion(args):
    rank_dir = 'trained_rank_npy\\%s\\batches_4' % args.network
    entropy_dir = 'trained_entropy_npy\\%s\\batches_4' % args.network
    rank_path = os.path.join(rank_dir, '%s_rank.npy' % args.network)
    entropy_path = os.path.join(entropy_dir, '%s_entropy.npy' % args.network)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(1, 10), copy=True)

    feature_maps_rank = np.array(np.load(rank_path)).reshape(-1, 1)
    scaled_rank = min_max_scaler.fit_transform(feature_maps_rank).reshape(-1).tolist()
    feature_maps_entropy = (np.array(np.load(entropy_path))).reshape(-1, 1)
    scaled_entropy = min_max_scaler.fit_transform(feature_maps_entropy).reshape(-1).tolist()

    mul = np.array([a * b for a, b in zip(scaled_rank, scaled_entropy)]).reshape(-1, 1)
    scaled_mul = min_max_scaler.fit_transform(mul).reshape(-1).tolist()

    prune_channels = []
    flags = [0, 1, 2]
    if args.network == 'resnet20':
        prune_channels = [0] * 18
        layer_wise_assign_prune_channels(prune_channels,
                                         layer1_prune=flags[0],
                                         layer2_prune=flags[1],
                                         layer3_prune=flags[2],
                                         num_block=3)
    elif args.network == 'resnet32':
        prune_channels = [0] * 30
        layer_wise_assign_prune_channels(prune_channels,
                                         layer1_prune=flags[0],
                                         layer2_prune=flags[1],
                                         layer3_prune=flags[2],
                                         num_block=5)
    elif args.network == 'resnet56':
        prune_channels = [0] * 54
        layer_wise_assign_prune_channels(prune_channels,
                                         layer1_prune=flags[0],
                                         layer2_prune=flags[1],
                                         layer3_prune=flags[2],
                                         num_block=9)
    elif args.network == 'resnet110':
        prune_channels = [0] * 108
        layer_wise_assign_prune_channels(prune_channels,
                                         layer1_prune=flags[0],
                                         layer2_prune=flags[1],
                                         layer3_prune=flags[2],
                                         num_block=18)

    stage = [0] * 3
    layer_num_0, layer_num_1, layer_num_2, = 0, 0, 0
    for i in range(len(scaled_mul)):
        if prune_channels[i] == flags[0]:
            stage[0] += scaled_mul[i]
            layer_num_0 += 1
        elif prune_channels[i] == flags[1]:
            stage[1] += scaled_mul[i]
            layer_num_1 += 1
        elif prune_channels[i] == flags[2]:
            stage[2] += scaled_mul[i]
            layer_num_2 += 1
    stage[0] /= layer_num_0
    stage[1] /= layer_num_1
    stage[2] /= layer_num_2

    # plot
    plt.style.use('seaborn-ticks')
    plt.rc('font', family='Times New Roman')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig_path = os.path.join(args.path, '%s_fusion.eps' % args.network)

    ret1 = plt.bar(0.5, stage[0], color="#426EB4", width=0.4)
    ret2 = plt.bar(1.5, stage[1], color="#3A2885", width=0.4)
    ret3 = plt.bar(2.5, stage[2], color="#00A4AC", width=0.4)
    plt.legend((ret1, ret2, ret3), ('Stage-1', 'Stage-2', 'Stage-3'), loc='upper right', prop={'size': 22})

    plt.xlim(0, None)

    plt.tick_params(labelsize=18)

    plt.xticks([0.5, 1.5, 2.5], [1, 2, 3])

    ax.set_xlabel("Stage", fontsize=26)
    ax.set_ylabel("Fusion Value", fontsize=26)

    plt.savefig(save_fig_path, format='eps', bbox_inches='tight', pad_inches=0)

    plt.close()


def main():
    args = load()

    plot_rank_entropy(args)
    plot_fusion(args)


if __name__ == '__main__':
    main()
