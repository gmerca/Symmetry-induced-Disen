from torch_geometric.data import DataLoader

from dataset.watts_strogatz import WattsStrogatz
from metrics.aggregator import MetricAggregator
from utils.CustomDataSet import SelectGraph
from utils.train_utils import load_model_result, train_cf, get_pyg_dataset, select_model
from classification.Classifier import MLP
import torch
import argparse


def main(arg):
    device = torch.device(arg.device)

    num_epoch = arg.e
    batch_size = arg.batch

    data_set = get_pyg_dataset(arg)
    # args.n_train = len(data_set) * 80 // 100
    # args.n_test = len(data_set) - args.n_train
    input_size = data_set.num_features
    # num_classes = data_set.num_classes
    shapes = list(map(int, arg.shapes.split(",")))
    # train_set = DataLoader(data_set[arg.n_skip:arg.n_skip + arg.n_train], batch_size=batch_size, shuffle=False)
    # test_set = DataLoader(data_set[arg.n_skip + arg.n_train:arg.n_skip + arg.n_train + arg.n_test], batch_size=batch_size, shuffle=False)

    model = select_model(arg, input_size, shapes, device)

    model.load_state_dict(torch.load(arg.model_dir + arg.m + ".ckpt"), strict=True)
    log_list = MetricAggregator(None, data_set, model, arg.points, False, 10, ntrue_actions=10, final=True,
                                fixed_shape=False, path=arg.model_dir, device=device, model_name=arg.m)()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Global_Dict generator')

    # for eval
    parser.add_argument('--task', type=str, default='eval', help="train / eval")
    parser.add_argument('--points', type=int, default=4, help="number of points")

    # loading model
    parser.add_argument('--group', type=int, default=100, help="group size")
    parser.add_argument('--sub_space', type=int, default=10, help="sub-space size")
    parser.add_argument('--hes', type=int, default=40, help="hyperparam for hessian penalty")

    # for compression model, same as train_compression.py
    parser.add_argument('--m', type=str, default='MIAGAE', help="model name")
    parser.add_argument('--device', type=str, default='cpu', help="cuda / cpu")
    parser.add_argument('--model_dir', type=str, default="data/model/", help="path to save model")
    parser.add_argument('--k', type=int, default=2, help="number of kernels")
    parser.add_argument('--depth', type=int, default=3, help="depth of encoder and decoder")
    parser.add_argument('--c_rate', type=float, default=0.8, help="compression ratio for each layer of encoder")
    parser.add_argument('--shapes', type=str, default="64,64,64", help="shape of each layer in encoder")

    # for classifier
    parser.add_argument('--d', type=str, default='WattsStrogatz', help="dataset name")
    parser.add_argument('--n_skip', type=int, default=0, help="skip some number of samples")
    parser.add_argument('--n_train', type=int, default=2000, help="number of samples for train set")
    parser.add_argument('--n_test', type=int, default=2000, help="number of samples for test set")
    parser.add_argument('--batch', type=int, default=1024, help="batch size")
    parser.add_argument('--e', type=int, default=1, help="number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--hidden', type=int, default=256, help="shape of each layer in encoder")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout rate")
    args = parser.parse_args()
    main(args)
