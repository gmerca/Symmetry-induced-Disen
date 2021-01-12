import os

import pandas as pd
from torch_geometric.loader import DataLoader

from classification.Classifier import MLP
from metrics.aggregator import MetricAggregator
from utils.train_utils import train_cp, select_model, get_pyg_dataset, setup, run_demo, train_cf, load_model_result, logging
import torch
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP


def main(rank, world_size, arg):
    # print(f"Running basic DDP example on rank {rank} ws {world_size}.")
    if world_size > 0:
        setup(rank, world_size)

    batch_size = arg.batch

    data_set = get_pyg_dataset(arg, rank)
    input_size = data_set.num_features
    shapes = list(map(int, arg.shapes.split(",")))
    cols = ["fac", "hig", "mig", "dci", "fl"]

    # if arg.d in ["WattsStrogatz", "ErdosRenyi"]:
    # arg.n_train = len(data_set) * 80 // 100
    # arg.n_test = len(data_set) - arg.n_train

    train_set_cp = DataLoader(data_set[:arg.n_train], batch_size=batch_size, shuffle=True)
    test_set_cp = DataLoader(data_set[arg.n_train:arg.n_train + arg.n_test], batch_size=batch_size, shuffle=False)

    train_set_cf = DataLoader(data_set[arg.n_skip:arg.n_skip + arg.n_train_cf], batch_size=batch_size, shuffle=False)
    test_set_cf = DataLoader(data_set[arg.n_skip + arg.n_train_cf:arg.n_skip + arg.n_train_cf + arg.n_test_cf], batch_size=batch_size, shuffle=False)

    def train_and_eval(run):
        model = select_model(arg, input_size, shapes, rank)
        optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001)
        trained = train_cp(model, optimizer1, rank, train_set_cp, test_set_cp, arg.num_epoch_cp, arg.model_dir, arg.m, arg.d)
        res = MetricAggregator(None, data_set, trained, arg.points, False, 10, ntrue_actions=10, final=True,
                               fixed_shape=False, path=arg.model_dir, device=rank, model_name=arg.m)()
        print(res)

        # group1, group2 = load_model_result(trained, train_set_cf, test_set_cf, rank)
        # input_size2 = group1[0].shape[1]
        # num_classes = data_set.num_classes
        # c_model = MLP(input_size2, arg.hidden, num_classes, arg.dropout)
        # c_model.to(rank)
        # c_model = DDP(c_model, device_ids=[rank]) if torch.cuda.is_available() else c_model
        # optimizer2 = torch.optim.Adam(c_model.parameters(), lr=0.01)
        # train_cf(c_model, optimizer2, rank, train_set_cf, test_set_cf, arg.num_epoch_cf, group1, group2, arg.model_dir, arg.m, arg.d)

    def save_df(df, name, exp):
        # if rank == 0 or rank == "cpu":
        logging("\n"+str(df.head(len(df))), os.path.join(arg.model_dir, 'eval.txt'), rank)
        df.to_csv(os.path.join(arg.model_dir, f'{exp}_{name}.csv'))
        # ax1 = df.plot.line()
        # fig = ax1.get_figure()
        # fig.savefig(os.path.join(arg.model_dir, f'{exp}_{name}.pdf'))

    def exp_disen(par_name, par, par_vals):
        """runs ablation study on single hyper-parameter"""
        default_ = par
        succeded = []
        file = os.path.join(arg.model_dir, 'log.txt')
        df = pd.DataFrame(columns=cols)
        for par in par_vals:
            logging(f"exp {par_name}: {par}", file, rank)
            arg.__setattr__(par_name, par)
            try:
                res = train_and_eval()
                df.loc[len(df)] = res
                succeded.append(par)
            except:
                logging("error on {}={}".format(par_name, par), file, rank)
        df[par_name] = succeded
        df = df.set_index(par_name)
        save_df(df, "default", par_name)
        arg.__setattr__(par_name, default_)

    def exp_cls(par_name, par, par_vals):
        """runs ablation study on single hyper-parameter"""
        default_ = par
        for par in par_vals:
            arg.__setattr__(par_name, par)
            train_and_eval()
        arg.__setattr__(par_name, default_)

    # arg.depth = 4
    # arg.k = 2
    # for arg.m in ['MIAGAE', 'GCN', 'GAT']:
    #     train_and_eval()

    # arg.depth = 1
    # arg.k = 1
    # arg.sub_space=10
    # train_and_eval()

    # for arg.depth in [1, 2, 3, 4, 5]:
    #     train_and_eval()
    # arg.cond = True
    # exp_disen("sub_space", arg.sub_space, [2, 3, 4, 5, 10])
    # exp_disen("group", arg.group, [36, 49, 81, 100])
    # exp_disen("hy_hes", arg.hy_hes, [0, 20, 40])
    # exp_disen("beta", arg.beta, [0.1, 0.4, 0.6, 1])
    # exp_disen("hy_commute", arg.hy_commute, [0, 5, 10, 20, 25])

    # for i in range(10):
    train_and_eval(0)
    # exp_cls("m", arg.m, ["Lie"])
    # exp_cls("m", arg.m, ["MIAGAE", "GCN", "GAT"])

    # for arg.cond in [False, True]:
    #     exp_cls("group", arg.group, [9, 16, 81])
    #     exp_cls("beta", arg.beta, [0.1, 0.2, 0.4, 0.5])
    #     exp_cls("hy_hes", arg.hy_hes, [0, 1, 5, 10, 15])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Global_Dict generator')

    parser.add_argument('--points', type=int, default=4, help="number of points")

    # lie group VAE
    parser.add_argument('--group', type=int, default=49, help="group size - 25, 64, 81, 100")
    parser.add_argument('--sub_space', type=int, default=3, help="sub-space size - 5, 10, 20")
    parser.add_argument('--hy_hes', type=float, default=40, help="hyperparam for hessian penalty - 5, 20, 40")
    parser.add_argument('--cond', action='store_true', default=False, help='conditional disen - True/False')
    parser.add_argument('--beta', type=float, default=1, help="beta - 0.2, 0.4, 0.5, 0.7, 0.8, 1")
    parser.add_argument('--cap', type=int, default=None, help="KL capacity")
    parser.add_argument('--loss', type=str, default='mse', help="mse / l2 / bce")
    parser.add_argument('--hy_rec', type=float, default=0.1, help="group rec loss penalty")
    parser.add_argument('--hy_commute', type=float, default=0, help="commutative loss penalty")

    parser.add_argument('--d', type=str, default='WattsStrogatz', help="dataset name")
    parser.add_argument('--m', type=str, default='MIAGAE', help="model name")
    parser.add_argument('--device', type=str, default='cuda', help="cuda / cpu")
    parser.add_argument('--batch', type=int, default=512, help="batch size")
    parser.add_argument('--num_epoch_cp', type=int, default=3, help="number of epochs")
    parser.add_argument('--num_epoch_cf', type=int, default=3, help="number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--model_dir', type=str, default="data/model/", help="path to save model")

    parser.add_argument('--n_skip', type=int, default=2000, help="skip some number of samples")
    parser.add_argument('--n_train', type=int, default=3000, help="number of samples for train set")
    parser.add_argument('--n_test', type=int, default=1000, help="number of samples for test set")
    parser.add_argument('--n_train_cf', type=int, default=100, help="number of samples for train set")
    parser.add_argument('--n_test_cf', type=int, default=100, help="number of samples for test set")
    parser.add_argument('--hidden', type=int, default=256, help="shape of each layer in encoder")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout rate")

    parser.add_argument('--k', type=int, default=2, help="number of kernels")
    parser.add_argument('--depth', type=int, default=3, help="depth of encoder and decoder")
    parser.add_argument('--c_rate', type=float, default=0.8, help="compression ratio for each layer of encoder")
    parser.add_argument('--shapes', type=str, default="64,64,64", help="shape of each layer in encoder")
    args = parser.parse_args()
    if torch.cuda.is_available():
        run_demo(main, torch.cuda.device_count(), args)
    else:
        main("cpu", 0, args)
