from typing import Union, Literal

import torch
import torch.nn as nn


def seed_weights(weights: list, seed: int) -> None:
    """Seed the weights of a list of nn.Parameter objects."""
    for i, weight in enumerate(weights):
        torch.manual_seed(seed + i)
        torch.nn.init.xavier_normal_(weight)


class StandardConv(nn.Conv2d):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def seed(self, s):
        seed_weights([self.weight], s)
        return self


class StandardLinear(nn.Linear):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def seed(self, s):
        seed_weights([self.weight], s)
        return self


class SimplexLayer:
    def __init__(self, init_weight: torch.Tensor, num_endpoints: int, seed: int):
        self.num_endpoints = num_endpoints
        self._alphas = tuple([1 / num_endpoints for _ in range(num_endpoints)])  # set by the train() method each round
        self._weights = nn.ParameterList([_initialize_weight(init_weight, seed + i) for i in range(num_endpoints)])

    @property
    def weight(self) -> nn.Parameter:
        return sum(alpha * weight for alpha, weight in zip(self._alphas, self._weights))

    def set_alphas(self, alphas: Union[tuple[float], Literal["center"]]):
        if len(alphas) == len(self._weights):
            self._alphas = alphas
        else:
            raise ValueError(f"alphas must match number of simplex endpoints ({self.num_endpoints})")


class SimplexLinear(nn.Linear, SimplexLayer):
    def __init__(self, num_endpoints: int, seed: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SimplexLayer.__init__(self, init_weight=self.weight, num_endpoints=num_endpoints, seed=seed)


class SimplexConv(nn.Conv2d, SimplexLayer):
    def __init__(self, num_endpoints: int, seed: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SimplexLayer.__init__(self, init_weight=self.weight, num_endpoints=num_endpoints, seed=seed)


def _initialize_weight(init_weight: torch.Tensor, seed: int) -> nn.Parameter:
    weight = nn.Parameter(torch.zeros_like(init_weight))
    torch.manual_seed(seed)
    torch.nn.init.xavier_normal_(weight)
    return weight
