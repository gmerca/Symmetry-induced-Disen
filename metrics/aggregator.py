import os
from datetime import datetime

from metrics import *
import torch
import warnings
from utils.train_utils import logging, lie_model_name, get_now, delta_time


class MetricAggregator:
    def __init__(self, ds, val_ds, model, num_points=1000, pair_ds=True, nactions=4, ntrue_actions=4, final=False,
                 nindep_epochs=30, fixed_shape=True, verbose=True, path="", device="", model_name=""):
        """ Helper class to compute disentanglement metrics

        Args:
            ds (Dataset): torch Dataset object on which to train metrics which need training
            val_ds (Dataset): torch Dataset object on which to evaluate metrics
            num_points (int): Number of points to use in metric calculation
            model (VAE): PyTorch model to evaluate
            pair_ds (bool): If True expect the dataset to output symmetry paired images
            nactions (int): The number of actions/symmetries to expect from RGrVAE/ForwardVAE
            ntrue_actions (int): The true number of actions
            final (bool): If True also evaluate the true independence
            nindep_epochs (int): Number of epochs to train independence representations for
            fixed_shape (bool): If fix shape in dsprites.
            verbose (bool): If True print verbosely
        """
        self.ds = ds
        self.num_points = num_points
        self.model = model
        self.paired = pair_ds
        self.val_ds = val_ds
        self.nactions = nactions
        self.final = final
        self.ntrue_actions = ntrue_actions
        self.nindep_epochs = nindep_epochs
        self.fixed_shape = fixed_shape
        self.verbose = verbose
        self.device = device
        log_file = os.path.join(path, 'eval.txt')
        self.log_file = log_file
        self.metrics = self._init_metrics(num_points)

        # log_str = f"\n{model_name} on {num_points} points\n"
        if model_name == "Lie":
            logging(f"\nDisen for {lie_model_name(model)} on {num_points} points", log_file, device)
        # logging(str(model), log_file)

    def _init_metrics(self, pts):
        fac = FactorVAEMetric(self.val_ds, pts * 10, pts * 5, 64, self.paired, self.fixed_shape, pts * 10, self.device)
        hig = BetaVAEMetric(self.val_ds, pts, 5, self.paired, self.fixed_shape, self.device)
        mig = MigMetric(self.val_ds, pts, self.paired, self.device)
        dci = DciMetric(self.val_ds, pts, self.paired, self.device)
        # mod = Modularity(self.val_ds, pts, self.paired, self.device)
        # sap = SapMetric(self.val_ds, pts, self.paired, self.device)
        # unsup = UnsupervisedMetrics(self.val_ds, pts, self.paired, self.device)  # comp expensive
        fl = FLMetric(self.val_ds, pts, self.paired, self.device)
        # ds = Downstream(self.val_ds, num_points=1000, paired=self.paired)  - check how to make this

        metrics = [fac, hig, mig, dci, fl]  #unsup sap, mod
        # metrics = [mig]
        return metrics

    def __call__(self):
        import gc
        with torch.no_grad():
            outputs = []
            times = []
            es = get_now(self.device)
            for metric in self.metrics:
                start = get_now(self.device)
                res = metric(self.model)
                end = get_now(self.device)
                if self.device == 0:
                    times.append(delta_time(end, start, self.device))
                # logging(str(res), self.log_file)
                outputs.append(res)
                gc.collect()
            ee = get_now(self.device)
            tot = delta_time(ee, es, self.device)
            multi = " ".join(times)
            names = [(n.__module__.split(".")[1]) for n in self.metrics]
            [logging("{}:{:.3f}".format(n, o), self.log_file, self.device) for n, o in zip(names, outputs)]
            logging("Eval Time: {} min, broken into {}".format(tot, multi), self.log_file, self.device)
            return outputs

