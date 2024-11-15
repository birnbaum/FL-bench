from argparse import ArgumentParser
from copy import deepcopy
from typing import Union, Literal

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn import decomposition

from src.client.floco import FlocoClient
from src.server.fedavg import FedAvgServer
from src.utils.constants import NUM_CLASSES
from src.utils.models import MODELS, DecoupledModel
from src.utils.tools import Namespace


class FlocoServer(FedAvgServer):

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--num_endpoints", type=int, default=1)  # TODO improve terminology
        parser.add_argument("--tau", type=int, default=100)  # TODO improve terminology
        parser.add_argument("--rho", type=float, default=0.1)  # TODO improve terminology
        parser.add_argument("--finetune_region", type=str, default='simplex_center')
        parser.add_argument("--evaluate_region", type=str, default='simplex_center')

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
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.model = SimplexModel(self.args)
        self.model.check_and_preprocess(self.args)
        self.init_trainer(FlocoClient)
        self.projected_clients = None

        if self.args.floco.pers_epoch > 0:
            self.clients_personalized_model_params = {
                i: deepcopy(self.model.state_dict()) for i in self.train_clients
            }

    def train_one_round(self):
        if self.args.floco.tau == self.current_epoch:
            print("Projecting gradients ... ")
            selected_clients = self.selected_clients  # save selected clients
            self.selected_clients = self.train_clients  # collect gradients
            client_packages = self.trainer.train()
            self.projected_clients = project_clients(
                client_packages, self.args.floco.endpoints, self.return_diff
            )
            self.selected_clients = selected_clients  # restore selected clients

        client_packages = self.trainer.train()
        if self.args.floco.pers_epoch > 0:
            for client_id in self.selected_clients:
                self.clients_personalized_model_params[client_id] = client_packages[
                    client_id
                ]["personalized_model_params"]
        self.aggregate(client_packages)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        if self.projected_clients is None:
            server_package["sample_from"] = (
                "simplex_center" if self.testing else "simplex_uniform"
            )
            server_package["subregion_parameters"] = None
        else:
            server_package["sample_from"] = (
                "subregion_center" if self.testing else "region_uniform"
            )
            server_package["subregion_parameters"] = (
                self.projected_clients[client_id],
                self.args.floco.rho,
            )
        if self.args.floco.pers_epoch > 0:
            server_package["personalized_model_params"] = (
                self.clients_personalized_model_params[client_id]
            )
        return server_package


class SimplexModel(DecoupledModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        base_model = MODELS[self.args.model.name](
            dataset=self.args.dataset.name,
            pretrained=self.args.model.use_torchvision_pretrained_weights,
        )
        self.base = base_model.base
        self.classifier = SimplexLinear(
            endpoints=self.args.floco.endpoints,
            in_features=base_model.classifier.in_features,
            out_features=NUM_CLASSES[self.args.dataset.name],
            bias=True,
            seed=self.args.common.seed,
        )
        self.sample_from = "simplex_center"
        self.subregion_parameters = None

    def forward(self, x):
        if self.sample_from == "simplex_center":
            alphas = np.ones(self.args.floco.endpoints) / np.ones(self.args.floco.endpoints).sum()
        else:
            center, radius = self.subregion_parameters
            alphas = sample_L1_ball(center, radius, 1)
        self.classifier.set_alphas(alphas)
        return super().forward(x)


class SimplexLinear(torch.nn.Linear):
    def __init__(self, endpoints: int, seed: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoints = endpoints
        self._alphas = tuple([1 / endpoints for _ in range(endpoints)])
        self._weights = torch.nn.ParameterList(
            [_initialize_weight(self.weight, seed + i) for i in range(endpoints)]
        )

    @property
    def weight(self) -> torch.nn.Parameter:
        return sum(alpha * weight for alpha, weight in zip(self._alphas, self._weights))

    def set_alphas(self, alphas: Union[tuple[float], Literal["center"]]):
        assert len(alphas) == len(self._weights)
        self._alphas = alphas


def project_clients(client_packages, endpoints, return_diff):
    model_grad_type = "model_params_diff" if return_diff else "regular_model_params"
    gradient_dict = {i: None for i in range(len(client_packages))}  # init sorted dict
    for client_id, package in client_packages.items():
        gradient_dict[client_id] = np.concatenate([
            v.cpu().numpy().flatten()
            for k, v in package[model_grad_type].items()
            if "classifier._weights" in k
        ])
    client_stats = np.array(list(gradient_dict.values()))
    pca_stats = decomposition.PCA(n_components=endpoints).fit_transform(client_stats)

    # Find optimal projection
    lowest_log_energy = np.inf
    best_projection = None
    for i, z in enumerate(np.linspace(1e-4, 1, 1000)):
        projection = _project_client(pca_stats, z=z)
        projection /= projection.sum(axis=1, keepdims=True)
        log_energy = _riesz_s_energy(projection)
        if log_energy not in [-np.inf, np.inf] and log_energy < lowest_log_energy:
            lowest_log_energy = log_energy
            best_projection = projection
    return best_projection


def _project_client(V, z):  # TODO what is V?
    """Projection of x onto the simplex, scaled by z.

    P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    """
    n_features = V.shape[1]
    U = np.sort(V, axis=1)[:, ::-1]
    z = np.ones(len(V)) * z
    cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
    ind = np.arange(n_features) + 1
    cond = U - cssv / ind > 0
    rho = np.count_nonzero(cond, axis=1)
    theta = cssv[np.arange(len(V)), rho - 1] / rho
    return np.maximum(V - theta[:, np.newaxis], 0)


def _riesz_s_energy(simplex_points):
    diff = simplex_points[:, None] - simplex_points[None, :]
    distance = np.sqrt((diff**2).sum(axis=2))
    np.fill_diagonal(distance, np.inf)
    epsilon = 1e-4  # epsilon is the smallest distance possible to avoid overflow during gradient calculation
    distance[distance < epsilon] = epsilon
    # select only upper triangular matrix to have each mutual distance once
    mutual_dist = distance[np.triu_indices(len(simplex_points), 1)]
    mutual_dist[np.argwhere(mutual_dist == 0).flatten()] = epsilon
    energies = (1 / mutual_dist ** 2)  # calculate energy by summing up the squared distances
    energy = energies[~np.isnan(energies)].sum()
    log_energy = -np.log(len(mutual_dist)) + np.log(energy)
    return log_energy


def sample_L1_ball(center, radius, num_samples):
    dim = len(center)
    samples = np.zeros((num_samples, dim))
    # Generate a point on the surface of the L1 unit ball
    u = np.random.uniform(-1, 1, dim)
    u = np.sign(u) * (np.abs(u) / np.sum(np.abs(u)))
    # Scale the point to fit within the radius
    r = np.random.uniform(0, radius)
    samples = center + r * u
    return samples


def _initialize_weight(init_weight: torch.Tensor, seed: int) -> torch.nn.Parameter:
    weight = torch.nn.Parameter(torch.zeros_like(init_weight))
    torch.manual_seed(seed)
    torch.nn.init.xavier_normal_(weight)
    return weight
