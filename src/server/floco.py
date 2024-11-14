from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn import decomposition
from src.client.floco import FlocoClient
from src.simplex import SolutionSimplex, projection_simplex, compute_riesz_s_energy

from src.server.fedavg import FedAvgServer
from src.utils.tools import Namespace


class FlocoServer(FedAvgServer):

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--num_endpoints", type=int, default=1)  # TODO improve terminology
        parser.add_argument("--tau", type=int, default=100)  # TODO improve terminology
        parser.add_argument("--rho", type=float, default=0.1)  # TODO improve terminology
        parser.add_argument("--finetune_region", type=str, default='sc')
        parser.add_argument("--evaluate_region", type=str, default='sc')

        # Floco+ (only used if pers_epoch > 0)
        parser.add_argument("--pers_epoch", type=int, default=0)
        parser.add_argument("--lamda", type=float, default=1)

        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "Floco",
        unique_model=False,
        use_fedavg_client_cls=True,
        return_diff=False,
    ):
        super().__init__(args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(FlocoClient)
        self.solution_simplex = SolutionSimplex(args=self.args)
        self.client_gradient_dict = {}  # TODO move this to train_one_round?

        if self.args.pers_epoch > 0:
            self.clients_personalized_model_params = {i: deepcopy(self.model.state_dict()) for i in self.train_clients}

    def train_one_round(self):
        if self.args.floco.tau == self.current_epoch:
            unsorted_client_statistics = np.array([v for _, v in self.client_gradient_dict.items()])
            unsorted_client_ids = np.array([int(k) for k in self.client_gradient_dict.keys()])
            sorted_args = np.argsort(unsorted_client_ids)
            client_statistics = unsorted_client_statistics[sorted_args]

            # Projection
            print('Projecting gradients ... ')

            client_statistics = decomposition.PCA(n_components=self.args.floco.num_endpoints).fit_transform(client_statistics)
            print('... finished PCA')

            # Offset z optimization
            statistics_over_z = []
            energies_over_z = []
            best_z = None
            last_log_energy = np.inf
            z_grid = np.linspace(1e-4, 1, 1000)
            for i, z in enumerate(z_grid):
                # 2. Optimized Simplex projection
                final_client_statistics = projection_simplex(client_statistics, z=z, axis=1)
                final_client_statistics /= final_client_statistics.sum(1).reshape(-1, 1)
                statistics_over_z.append(final_client_statistics)
                _, log_energy = compute_riesz_s_energy(final_client_statistics, d=2)
                if log_energy not in [-np.inf, np.inf]:
                    energies_over_z.append(log_energy)
                    if log_energy < last_log_energy:
                        best_z = i
                        last_log_energy = log_energy
            print('... finished z optimization')
            client_statistics = np.array(statistics_over_z)[best_z]

            ### !!!!!
            self.solution_simplex.set_solution_simplex_regions(
                projected_points=client_statistics,
                rho=self.args.floco.rho,
            )
            self.centers = [simplex_region.center_simplex for simplex_region in self.solution_simplex.simplex_regions]
            ### !!!!!


        elif self.args.floco.tau == (self.current_epoch + 1):
            # Sample all clients to get most up to date gradient, or simplex information
            self.last_selected_client_cids = [client_id for client_id in self.selected_clients]
            self.selected_clients = self.train_clients

        client_packages = self.trainer.train()

        if self.args.pers_epoch > 0:
            for client_id in self.selected_clients:
                self.clients_personalized_model_params[client_id] = client_packages[client_id]["personalized_model_params"]

        self.aggregate(client_packages)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        if self.current_epoch < self.args.floco.tau:
            server_package["sample_from"] = "simplex_center" if self.testing else "simplex_uniform"
        else:
            server_package["sample_from"] = "region_center" if self.testing else "region_uniform"

        if self.args.pers_epoch > 0:
            server_package["personalized_model_params"] = self.clients_personalized_model_params[client_id]
        return server_package

    @torch.no_grad()
    def aggregate(self, clients_package: OrderedDict[int, dict[str, Any]]):
        weights_results = [package for package in clients_package.values()]
        if self.args.floco.tau == (self.current_epoch + 1):
            weight_result_cids = np.array([client_id for client_id in clients_package.keys()])

            # Sort client gradients/losses for later clustering
            new_weight_results = []
            for cid in self.last_selected_client_cids:
                arg_id = np.argwhere(cid == np.array(weight_result_cids))[0][0]
                client_weight_results = weights_results[arg_id]
                new_weight_results.append(client_weight_results)
            weights_results = new_weight_results

            # Save all client gradients/losses for later clustering
            if self.return_diff:
                model_grad_type = "model_params_diff"
            else:
                model_grad_type = "regular_model_params"

            for client_id, package in clients_package.items():
                w = [val.cpu().numpy() for _, val in package[model_grad_type].items()]
                client_grads = [w[-i].flatten() for i in range(1, self.args.floco.num_endpoints + 1)]
                client_grads = np.concatenate(client_grads)
                self.client_gradient_dict[client_id] = client_grads

        ###

        client_weights = [package["weight"] for package in weights_results]
        weights = torch.tensor(client_weights) / sum(client_weights)

        if self.return_diff:  # inputs are model params diff
            for name, global_param in self.public_model_params.items():
                diffs = torch.stack(
                    [
                        package["model_params_diff"][name]
                        for package in weights_results
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(diffs * weights, dim=-1)
                self.public_model_params[name].data -= aggregated
        else:
            for name, global_param in self.public_model_params.items():
                client_params = torch.stack(
                    [
                        package["regular_model_params"][name]
                        for package in weights_results
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(client_params * weights, dim=-1)
                global_param.data = aggregated
