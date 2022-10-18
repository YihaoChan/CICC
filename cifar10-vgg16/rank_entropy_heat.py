import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os
import numpy as np


def load():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='D:\\programme\\NIPS\\images\\rank_entropy\\heat')
    parser.add_argument('--type', type=str, default='entropy')
    parser.add_argument('--interval', type=int, default=2)

    args = parser.parse_args()

    return args


def stat(args):
    feature_maps_entropy = []

    batches_num = []

    for root, dirs, files in os.walk(os.path.join('./trained_%s_npy' % args.type)):
        for file in files:
            file_path = os.path.join(root, file)
            if 'batches_' in file_path:
                batches_num.append(
                    os.path.abspath(os.path.dirname(file_path) + os.path.sep + ".").split('batches_')[-1])
                entropy = np.array(np.load(file_path))
                feature_maps_entropy.append(entropy)

    return feature_maps_entropy, batches_num


def plot_rank_entropy(feature_maps_entropy, batches_num, args):
    sns.set(font="Times New Roman", font_scale=1)

    df = pd.DataFrame(feature_maps_entropy)
    df.columns = np.arange(len(feature_maps_entropy[0])) + 1
    df.index = batches_num

    ax = sns.heatmap(df, cmap='Spectral_r', linewidths=0.3, cbar=True)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    # plt.xlabel("Layer", fontsize=13)
    # plt.ylabel("Number of Batches", fontsize=13)
    # plt.title("%s per Layer" % (args.type.title()), fontsize=15)

    plt.xticks(np.arange(0.5, len(feature_maps_entropy[0]) + 0.5, args.interval),
               np.arange(1, len(feature_maps_entropy[0]) + 1, args.interval))

    if not os.path.exists(args.path):
        os.makedirs(args.path)
    save_fig_path = os.path.join(args.path, 'vgg16_%s_heat.eps' % args.type)
    plt.savefig(save_fig_path, format='eps', bbox_inches='tight', pad_inches=0)


def main():
    args = load()

    feature_maps_entropy, batches_num = stat(args)

    plot_rank_entropy(feature_maps_entropy, batches_num, args)


if __name__ == '__main__':
    main()
