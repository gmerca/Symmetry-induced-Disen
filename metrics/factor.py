"""Implementation of FactorVAE Metric.

Based on "Disentangling by Factorising" (https://arxiv.org/abs/1802.05983).
Implementation based on https://github.com/google-research/disentanglement_lib
"""

import numpy as np
import torch
from sklearn import linear_model
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class FactorVAEMetric:
    def __init__(self, ds, num_train=10000, num_eval=5000, bs=64, paired=False, fixed_shape=True, n_var_est=10000,
                 device="cpu"):
        """ FactorVAE Metric

        Args:
            ds (Dataset): torch dataset on which to evaluate
            num_train (int): Number of points to train on
            num_eval (int): Number of points to evaluate on
            bs (int): batch size
            paired (bool): If True expect the dataset to output symmetry paired images
            fixed_shape (bool): If fix shape in dsprites.
            n_var_est (int): Number of examples to estimate global variance.
        """
        super().__init__()
        self.ds = ds
        self.num_train = num_train
        self.num_eval = num_eval
        self.bs = bs
        self.paired = paired
        self.fixed_shape = fixed_shape
        self.n_var_est = n_var_est
        self.device = device

        if 'flatland' in str(type(self.ds)):
            self.num_factors = len(self.ds.latents_sizes)
        elif 'dsprites' in str(type(self.ds)):
            if self.fixed_shape:
                self.num_factors = len(self.ds.latents_sizes) - 1
            else:
                self.num_factors = len(self.ds.latents_sizes)
        else:
            self.num_factors = len(self.ds.latents_sizes)

    def __call__(self, pymodel):
        rep_fn = lambda x: pymodel(x.to(self.device))
        global_var = self._compute_variances(rep_fn, self.n_var_est)
        active_dims = self._prune_dims(global_var)
        scores_dict = {}

        if not active_dims.any():
            scores_dict["dmetric/fac_train"] = 0.
            scores_dict["dmetric/fac_eval"] = 0.
            scores_dict["dmetric/fac_num_act"] = 0
            return scores_dict

        train_votes = self._get_train_votes(rep_fn, self.bs, self.num_train,
                                            global_var, active_dims)
        # print('train_votes:', train_votes)
        classifier = np.argmax(train_votes, axis=0)
        other_index = np.arange(train_votes.shape[1])
        train_accuracy = np.sum(
            train_votes[classifier, other_index]) * 1. / np.sum(train_votes)

        eval_votes = self._get_train_votes(rep_fn, self.bs, self.num_eval,
                                           global_var, active_dims)
        # print('eval_votes:', eval_votes)
        eval_accuracy = np.sum(
            eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)
        scores_dict["dmetric/fac_train"] = "{:.4f}".format(train_accuracy)
        scores_dict["dmetric/fac_eval"] = "{:.4f}".format(eval_accuracy)
        scores_dict["dmetric/fac_num_act"] = "{:.4f}".format(active_dims.astype(int).sum())
        return eval_accuracy

    def _get_train_votes(self, rep_fn, bs, num_points, global_var, active_dims):
        votes = np.zeros((self.num_factors, global_var.shape[0]),
                         dtype=np.int64)
        for _ in tqdm(range(num_points)):
            factor_index, argmin = self._generate_training_sample(rep_fn, bs,
                                                             global_var, active_dims)
            votes[factor_index, argmin] += 1
        return votes

    def _generate_training_sample(self, rep_fn, bs, global_var, active_dims):
        # Select random coordinate to keep fixed.
        factor_index_metric = np.random.randint(self.num_factors)
        if 'dsprites' in str(type(self.ds)) and self.fixed_shape:
            factor_index = factor_index_metric + 1
        else:
            factor_index = factor_index_metric
        obs = []
        for i in range(bs):
            # Sample two mini batches of latent variables.
            factor = self.ds.sample_latent()
            # Fix the selected factor across mini-batch.
            if i == 0:
                fac_mem = factor[factor_index]
            else:
                factor[factor_index] = fac_mem
            # Obtain the observations.
            ob = self.ds.get_img_by_latent(factor)
            # if not torch.is_tensor(ob):
            #     ob = ob[0]
            obs.append(ob)
        # obs = torch.stack(obs)
        loader = DataLoader(obs, batch_size=len(obs), shuffle=False)
        for b in loader:
            _, reps, _, _, _ = rep_fn(b)
        reps = reps.cpu().numpy()
        local_variances = np.var(reps, axis=0, ddof=1)
        argmin = np.argmin(local_variances[active_dims] / global_var[active_dims])
        return factor_index_metric, argmin

    def _prune_dims(self, variances, threshold=1.9e-09):
        """Mask for dimensions collapsed to the prior."""
        scale_z = np.sqrt(variances)
        return scale_z >= threshold

    def _compute_variances(self, rep_fn,
                           bs,
                           eval_bs=64):
        obs = []
        for _ in tqdm(range(bs)):
            latent = self.ds.sample_latent()
            obs_i = self.ds.get_img_by_latent(latent).to(self.device)
            obs.append(obs_i)
        # obs = torch.stack(obs)
        reps = self._obtain_representation(obs, rep_fn, eval_bs)

        # assert reps.shape[0] // reps.shape[1] == bs
        return np.var(reps, axis=0, ddof=1)

    def _obtain_representation(self, obs, rep_fn, bs):
        reps = None
        num_points = len(obs)
        i = 0
        while i < num_points:
            num_points_iter = min(num_points - i, bs)
            cur_obs = obs[i:i + num_points_iter]
            # ds, _ = self.ds.collate(cur_obs)
            loader = DataLoader(cur_obs, batch_size=len(cur_obs), shuffle=False)
            if i == 0:
                _, reps, _, _, _ = rep_fn(loader._get_iterator().next())
                reps = reps.to(self.device).cpu().numpy()
            else:
                _, reps1, _, _, _ = rep_fn(loader._get_iterator().next())
                reps = np.vstack((reps, reps1.to(self.device).cpu().numpy()))
            i += num_points_iter
        return reps
