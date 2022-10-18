# -*- coding: UTF-8 -*-
import os
import argparse


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device-ids', type=int, nargs='+', default=[0, 1, 2, 3])

    parser.add_argument('--multi-gpu', action='store_true', default=False)

    parser.add_argument('--network', required=True)

    parser.add_argument('--train-flag', action='store_true',
                        help='flag for training network', default=False)

    parser.add_argument('--prune-flag', action='store_true',
                        help='flag for pruning network', default=False)

    parser.add_argument('--resume-prune-flag', action='store_true',
                        help='flag for resuming pruned model', default=False)

    parser.add_argument('--resume-prune-epoch', type=int,
                        help='number of epoch for resuming pruned model', default=1)

    parser.add_argument('--ablation-flag', action='store_true',
                        help='flag for ablation study', default=False)

    parser.add_argument('--ablation-layer', type=str,
                        help='number of epoch for resuming pruned model', default='layer1')

    parser.add_argument('--resume-prune-lr', type=float,
                        help='lr for resuming pruned model', default=0.001)

    parser.add_argument('--retrain-epoch', type=int,
                        help='number of epoch for retraining pruned network', default=90)

    parser.add_argument('--retrain-lr', type=float,
                        help='learning rate for retraining pruned network', default=0.01)

    parser.add_argument('--lr-adjust', type=int, default=30, help='number of epochs that change learning rate')

    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

    parser.add_argument('--dataset-path', type=str, default='/public/datasets/ILSVRC2012')

    parser.add_argument('--start-epoch', type=int,
                        help='start epoch for training network', default=0)

    parser.add_argument('--epoch', type=int,
                        help='number of epoch for training network', default=90)

    parser.add_argument('--batch-size', type=int,
                        help='batch size', default=256)

    parser.add_argument('--num-workers', type=int,
                        help='number of workers for data loader', default=16)

    parser.add_argument('--lr', type=float,
                        help='learning rate', default=0.1)

    parser.add_argument('--momentum', type=float,
                        help='momentum for optimizer', default=0.9)

    parser.add_argument('--weight-decay', type=float,
                        help='factor for weight decay in optimizer', default=1e-4)

    parser.add_argument('--print-freq', type=int,
                        help='print frequency during training', default=100)

    parser.add_argument('--finetune-epoch', type=int,
                        help='number of epoch for layer-wise finetune', default=2)

    parser.add_argument('--train-save-path', type=str,
                        help='train model save path', default='./trained_models')

    parser.add_argument('--prune-save-path', type=str,
                        help='pruned model save path', default='./pruned_models')

    parser.add_argument('--resume-save-path', type=str,
                        help='resume model save path', default='./resume_models')

    parser.add_argument('--ablation-save-path', type=str,
                        help='ablation study save path', default='./ablation_models')

    parser.add_argument('--sv-save-path', type=str, default='./trained_shap_values')

    parser.add_argument('--shap-value-path', type=str, default='shap_value.npy')

    parser.add_argument('--log-path', type=str, default='./logs')

    return parser


def get_parameter():
    parser = build_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    save_path = ''

    if args.train_flag:
        save_path = args.train_save_path

    if args.resume_prune_flag:
        save_path = args.resume_save_path

    if args.prune_flag:
        save_path = args.prune_save_path

    if args.ablation_flag:
        save_path = os.path.join(args.ablation_save_path, args.ablation_layer)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(args.sv_save_path):
        os.makedirs(args.sv_save_path)

    args.shap_value_path = "%s_%s" % (args.network, args.shap_value_path)

    return args
