from __future__ import division, print_function, absolute_import
import copy
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 10

# Grid resolution
RES = 200

import sys

sys.stdout = sys.stderr

import torch
import numpy as np

from src.simplex_layers import SimplexLayer


def map_simplex_to_cartesian(simplex_point):
    """
    Map a point from standard 2-simplex with vertices: (0,0,1), (0,1,0), (1,0,0),
    to cartesian triangle: triangle(ABC), A=(0,0), B=(1,0), C=(0,1)
    """
    return simplex_point[1:]


def map_cartesian_to_simplex(cartesian_point, res=RES):
    """
    Map a point from cartesian triangle (0,0), (0,1), (1,0) to stanard 2-simplex.
    Cartesian point represented as (x,y)
    """
    cartesian_point = np.array(cartesian_point)
    simplex_point = np.concatenate([
        np.array([res - sum(cartesian_point)]) / res,
        cartesian_point / res
    ])
    return simplex_point


# A class that defines a region in the solution simplex with a center, and coordinate mappings between the coordinates simplex <-> cartesian
class SimplexRegion:
    """
    Simplex Region/Clustering Region class. Is defined through its cluster center in both simplex & cartesian coordinates. Can sample from its region given certain rho parameter.
    """

    def __init__(
            self,
            region_id,
            center,
            rho
    ):
        self.region_id = region_id
        self.center_simplex = center
        self.center_cartesian = map_simplex_to_cartesian(self.center_simplex)
        # Radius of the ball to sample around from
        self.rho = rho

        # Presample alphas to accelerate training
        sampled_alphas = sample_L1_ball(self.center_simplex, self.rho, 100000)
        # Normalization, IMPORTANT!
        self.sampled_alphas = sampled_alphas / sampled_alphas.sum(axis=1).reshape(-1, 1)
        # Append center
        self.sampled_alphas = np.concatenate([self.sampled_alphas, [self.center_simplex]])
        self.alphas_cartesian = [map_simplex_to_cartesian(alpha) for alpha in self.sampled_alphas]

    def get_client_subregion(self):
        return self.sampled_alphas, self.alphas_cartesian

    def get_region_center_simplex(self):
        return self.center_simplex

    def get_region_center_cartesian(self):
        return self.center_cartesian


class SolutionSimplex:
    """
    Solution simplex that keeps track of all simplex regions.
    """

    def __init__(self, args):
        self.args = args

        if "flocop" in self.args:
            self.rho = self.args.flocop.rho
        else:
            self.rho = self.args.floco.rho

        if "flocop" in self.args:
            self.num_endpoints = self.args.flocop.num_endpoints
        else:
            self.num_endpoints = self.args.floco.num_endpoints

    def set_solution_simplex_regions(
            self, projected_points, rho
    ):

        self.simplex_regions = _compute_solution_simplex_(
            projected_points=projected_points,
            rho=self.rho
        )
        self.rho = rho
        self.client_to_simplex_region_mapping = {}
        for i, simplex_region in enumerate(self.simplex_regions):
            self.client_to_simplex_region_mapping[i] = simplex_region.region_id

    def get_client_subregion(self, client_id):
        # Get correct simplex region
        sampled_region_id = self.client_to_simplex_region_mapping[client_id]
        client_simplex_region = self.simplex_regions[sampled_region_id]
        sampled_alpha_simplex, sampled_alpha_cartesian = client_simplex_region.get_client_subregion()
        return sampled_alpha_simplex, sampled_alpha_cartesian, sampled_region_id

    def sample_uniform(self, client_id):
        # Get correct simplex region
        sampled_region_id = client_id
        alpha = np.random.exponential(scale=1.0, size=(100000, self.num_endpoints))
        sampled_alpha_simplex = alpha / alpha.sum(1).reshape(-1, 1)
        simplex_center = np.ones(self.num_endpoints) / np.ones(self.num_endpoints).sum()
        sampled_alpha_simplex = np.concatenate([sampled_alpha_simplex, [simplex_center]])
        sampled_alpha_cartesian = [map_simplex_to_cartesian(alpha) for alpha in sampled_alpha_simplex]
        return sampled_alpha_simplex, sampled_alpha_cartesian, sampled_region_id

    def get_client_center(self, client_id):
        sampled_region_id = self.client_to_simplex_region_mapping[client_id]
        alpha_simplex = self.simplex_regions[sampled_region_id].get_region_center_simplex()
        alpha_cartesian = self.simplex_regions[sampled_region_id].get_region_center_cartesian()
        return [alpha_simplex], alpha_cartesian, sampled_region_id

    def get_simplex_region_centers_cartesian(self):
        return [simplex_region.get_region_center_cartesian() for simplex_region in self.simplex_regions]

    def get_simplex_region_centers_simplex(self):
        return [simplex_region.get_region_center_simplex() for simplex_region in self.simplex_regions]


def _compute_solution_simplex_(projected_points, rho):
    simplex_regions = []
    for i, tmp_center in enumerate(projected_points):
        simplex_region = SimplexRegion(
            region_id=i,
            center=tmp_center,
            rho=rho
        )
        simplex_regions.append(simplex_region)
    return simplex_regions


def sample_L1_ball(center, radius, num_samples):
    dim = len(center)
    samples = np.zeros((num_samples, dim))
    for i in range(num_samples):
        # Generate a point on the surface of the L1 unit ball
        u = np.random.uniform(-1, 1, dim)
        u = np.sign(u) * (np.abs(u) / np.sum(np.abs(u)))
        # Scale the point to fit within the radius
        r = np.random.uniform(0, radius)
        samples[i] = center + r * u
    return samples


def set_net_alpha(net, alphas: tuple[float, ...]):
    for m in net.modules():
        if isinstance(m, SimplexLayer):
            m.set_alphas(alphas)





def sample_model_point_estimate(model, new_model, sampling_point):
    """
    Creates a new model with the same architecture as the input model,
    but replaces the last layer's ParameterList with a Linear layer using
    the (weighted) average of its weights.

    Parameters:
    model (nn.Module): The PyTorch model to be processed.

    Returns:
    nn.Module: A new model with a Linear layer in place of the ParameterList.
    """
    # Extract the last layer (assuming it's a fully connected layer with ParameterList)
    last_layer = copy.deepcopy(model.classifier._weights)
    last_layer_bias = copy.deepcopy(model.classifier.bias)

    # Check if last layer is nn.ParameterList
    if not isinstance(last_layer, torch.nn.ParameterList):
        raise TypeError("The last layer must be a ParameterList containing weight tensors.")

    # Compute the average of the tensors in the ParameterList
    final_weight = 0
    for factor, weight in zip(sampling_point, last_layer):
        final_weight += factor * weight

    # Replace the last layer in the new model with a Linear layer
    # Assuming the input features of the linear layer are the same as the last Parameter tensor
    in_features, out_features = final_weight.size()
    new_model.classifier = torch.nn.Linear(in_features, out_features, bias=True)

    # Set the new Linear layer's weights to the averaged weights
    with torch.no_grad():
        new_model.classifier.weight = torch.nn.Parameter(final_weight)
        new_model.classifier.bias = last_layer_bias

    return new_model
