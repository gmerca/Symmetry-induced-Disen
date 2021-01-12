import math
import time
from datetime import datetime

import dateutil.relativedelta
import torch
import os

from tqdm import tqdm

from dataset.alchemy import TencentAlchemyDataset
from dataset.erdos_renyi import ErdosRenyi
from dataset.factor_graphs import FactorGraphs
from dataset.watts_strogatz import WattsStrogatz
from utils.CustomDataSet import SelectGraph

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def load_model_result(model, train_set, test_set, device):
    x1, e1, batch1 = None, None, None
    for data in train_set:
        data = data.to(device)
        _, x1, e1, batch1, _ = model(data)
    # mean = torch.mean(x1, dim=0)
    # std = torch.std(x1, dim=0) + 1e-12
    # x1 = (x1 - mean) / std

    x2, e2, batch2 = None, None, None
    for data in test_set:
        data = data.to(device)
        _, x2, e2, batch2, _ = model(data)
    # x2 = (x2 - mean) / std

    return [x1.detach(), e1.detach(), batch1.detach()], \
           [x2.detach(), e2.detach(), batch2.detach()]


def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def run_demo(demo_fn, world_size, args):
    mp.spawn(demo_fn,
             args=(world_size, args,),
             nprocs=world_size,
             join=True)


def lie_model_name(model):
    li = list()
    mod = model.module if torch.cuda.is_available() else model
    li.append(f"g{mod.group_size}")
    li.append(f"s{mod.sub_space_size}")
    li.append(f"h{mod.hy_hes}")
    li.append(f"con{int(mod.cond_disen)}")
    li.append(f"com{mod.hy_commute}")
    li.append(f"l{mod.loss[0]}")
    li.append(f"cap{mod.capacity}")
    li.append(f"b{mod.beta}")
    li.append(f"d{mod.depth}")
    return "_".join(li)


def get_now(rank):
    if rank == 0 or rank == "cpu":
        return datetime.now()


def delta_time(end, start, rank):
    if rank == 0 or rank == "cpu":
        return "{:.2f}".format((end - start).total_seconds() / 60)


def train_cp(model, optimizer, rank, train_set, valid_set, num_epoch, path, m_name, ds_name):
    """train compression"""
    start = get_now(rank)
    log_file = os.path.join(path, 'log.txt')
    log_str = '\nTrain {} on {} '.format(m_name, ds_name)
    if m_name == "Lie":
        log_str += lie_model_name(model)
        m = model.module if torch.cuda.is_available() else model
        m_name += "_cond" if m.cond_disen else ""
    logging(log_str, log_file, rank)
    min_loss = math.inf
    best_model = None

    for e in tqdm(range(num_epoch)):
        tot_tr, tot_val, mse_val, group_val, kl_val, mse_tr, kl_tr, group_tr = 0, 0, 0, 0, 0, 0, 0, 0
        for data in train_set:
            optimizer.zero_grad()
            model.train()
            z, _, _, _, tr_loss = model(data)
            tr_loss["total"].backward()
            tot_tr += tr_loss["total"].item()
            mse_tr += tr_loss["rec"].item()
            kl_tr += tr_loss["kl"].item()
            group_tr += tr_loss["group"].item()
            optimizer.step()

        for data in valid_set:
            model.eval()
            z, _, _, _, val_loss = model(data)
            tot_val += val_loss["total"].item()
            mse_val += val_loss["rec"].item()
            kl_val += val_loss["kl"].item()
            group_val += val_loss["group"].item()

        tot_val, mse_val, kl_val, group_val = tot_val/len(valid_set), mse_val/len(valid_set), kl_val/len(valid_set), group_val/len(valid_set)
        tot_tr, mse_tr, kl_tr, group_tr = tot_tr/len(train_set), mse_tr/len(train_set), kl_tr/len(train_set), group_tr/len(train_set)

        if e % 15 == 0:
            log_str = 'Epoch {:3d}, '.format(e)
            tr_str = 'Tot:{:.3f}, MSE:{:.3f} KL:{:.3f} group:{:.3f}'.format(tot_tr, mse_tr, kl_tr, group_tr)
            val_str = 'Tot:{:.3f}, MSE:{:.3f} KL:{:.3f} group:{:.3f}'.format(tot_val, mse_val, kl_val, group_val)
            logging(f"{log_str}\t\t TR {tr_str}\t\tVAL {val_str}", log_file, rank)

        if tot_val < min_loss:
            min_loss = tot_val
            os.makedirs(path, exist_ok=True)
            best_model = model
    end = get_now(rank)
    tt = delta_time(end, start, rank)
    logging("Tot Loss = {:.3f}\t\tTrain Time: {} min".format(min_loss, tt), log_file, rank)
    torch.save(best_model.state_dict(), path + m_name + f"_{ds_name}" + ".ckpt")
    return best_model


def train_cf(model, optimizer, device, train_set, valid_set, num_epoch, group1, group2, path, m_name, ds_name):
    log_file = os.path.join(path, 'log.txt')
    # log_str = '\nTraining {}\t classifier on {}'.format(m_name, ds_name)
    # if device == 0 or device == "cpu":
    #     logging(log_str, log_file)
    best_acc = 0
    for e in tqdm(range(num_epoch)):
        c_loss = 0
        train_accuracy = 0
        total_num = 0
        for data in train_set:
            optimizer.zero_grad()
            model.training = True
            c = model(group1[0], group1[1], group1[2])
            label = data.y.long().to(device)
            pred = c.argmax(dim=1).to(device)
            total_num += label.shape[0]
            train_accuracy += (pred == label).sum().item()
            c_loss = torch.nn.CrossEntropyLoss()(c, label)
            c_loss.backward()
            optimizer.step()

        train_accuracy /= total_num

        test_accuracy = 0
        total_num = 0
        for data in valid_set:
            model.training = False
            c = model(group2[0], group2[1], group2[2])
            pred = c.argmax(dim=1).to(device)
            label = data.y.long().to(device)
            total_num += label.shape[0]
            test_accuracy += (pred == label).sum().item()

        test_accuracy /= total_num
        if test_accuracy > best_acc:
            best_acc = test_accuracy

        # if (device == 0 or device == "cpu") and e % 10 == 0:
        #     logging('Epoch {} | Train Loss {:.3f} Train Acc {:.3f} Test Acc: {:.3f}'.format(e, c_loss, train_accuracy, test_accuracy), log_file)

    logging(f"Best class acc {best_acc}", log_file, device)


def select_model(arg, input_size, shapes, device):
    """
    select model for training compression or load model for training classification
    """
    if arg.m == "MIAGAE":
        from classification.Graph_AE import Net
        model = Net(input_size, arg.k, arg.depth, [arg.c_rate] * arg.depth, shapes, device)
    elif arg.m == "GCN":
        from classification.GCN_AE import Net
        model = Net(input_size, arg.k, arg.depth, [arg.c_rate] * arg.depth, shapes, device)
    elif arg.m == "GAT":
        from classification.GAT_AE import Net
        model = Net(input_size, arg.k, arg.depth, [arg.c_rate] * arg.depth, shapes, device)
    elif arg.m == "Lie":
        from classification.Graph_LieVAE import Net
        model = Net(input_size, arg.k, arg.depth, [arg.c_rate] * arg.depth, shapes, device, arg.batch,
                    arg.group, arg.sub_space, arg.hy_hes, arg.cond, arg.beta, arg.cap, arg.loss, arg.hy_rec, arg.hy_commute)
    elif arg.m == "UNet":
        from classification.UNet import Net
        model = Net(input_size, arg.depth, arg.c_rate, shapes, device)
    elif arg.m == "Gpool":
        from classification.Gpool_model import Net
        model = Net(input_size, arg.depth, arg.c_rate, shapes, device)
    elif arg.m == "SAGpool":
        from classification.SAG_model import Net
        model = Net(input_size, arg.depth, [arg.c_rate] * arg.depth, shapes, device)
    else:
        print("model not found")
        return
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[device], find_unused_parameters=True) if torch.cuda.is_available() else model
    # ddp_model = model
    return ddp_model


def get_pyg_dataset(args, rank):
    """ load dataset """
    # print("loading data ")
    if args.d == "WattsStrogatz":
        if not torch.cuda.is_available():
            data_set = WattsStrogatz("data/WattsStrogatz/", prob=10, knn=10, mean=10, device=rank)
        else:
            data_set = WattsStrogatz("data/WattsStrogatz/", prob=20, num_nodes=6, mean=20, knn=5, device=rank, num_feat=32)
        if rank==0:
            print(data_set)
    elif args.d == "ErdosRenyi":
        data_set = ErdosRenyi("data/ErdosRenyi/", prob=5, logvar=5, mean=5, num_nodes=10, device=rank)
    elif args.d == "FactorGraphs":
        data_set = FactorGraphs("data/FactorGraphs/", num_graphs=500, device=rank)
    elif args.d == "Alchemy":
        data_set = TencentAlchemyDataset("data/Alchemy/")
    else:
        SelectGraph.data_name = args.d
        data_set = SelectGraph('data/' + SelectGraph.data_name)
    # print("done")
    return data_set


def logging(s, path, rank, print_=True):
    if rank == 0 or rank == "cpu":
        if print_:
            print(s)
        if path:
            with open(path, 'a+') as f:
                f.write(s + '\n')

