from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchmetrics

from pytorch_utils.exceptions import (
    NotMonotone,
    NotNonDecreasing,
    NotNonIncreasing,
)
from pytorch_utils.logging.loggers import (
    Logger,
    VoidLogger,
    use_loggers,
)

NamedTorchTensors = Mapping[str, torch.Tensor]
NamedTorchModules = Mapping[str, nn.Module]
NamedTorchMetrics = Mapping[str, torchmetrics.Metric]
BatchTorchTensors = Tuple[NamedTorchTensors, torch.Tensor, Optional[torch.Tensor]]


@dataclass(frozen=True, order=True)
class CategoricalFeatureEmbedding:
    """
    Collection that simplifies the constructor parameters.
    """

    feature_name: str
    nb_distinct_values: int
    embedding_size: int
    logger: Logger = field(default=VoidLogger(), repr=False, compare=False)

    def __post_init__(self):
        # Check attributes
        # ------------------------
        if self.nb_distinct_values < 0:
            raise ValueError("nb_distinct_values must be non-negative")
        if self.embedding_size < 0:
            raise ValueError("embedding_size must be non-negative")

        # Make this class comply with Loggable Protocol
        object.__setattr__(self, "loggers", {"logger": self.logger})

    @use_loggers("logger")
    def log(self, logger: Logger) -> None:
        params = asdict(self)
        del params["logger"]
        logger.log_params(params)


def get_embedding_size(
    nb_categories: int,
    multiplicative_factor: float = 1.6,
    power_exponent: float = 0.56,
    max_size: int = 600,
) -> int:
    """
    Determine empirically good embedding sizes (formula taken from fastai: https://docs.fast.ai/tabular.model.html).
    Parameters
    ----------
    nb_categories: int
        number of categories
    max_size: (int, optional)
        maximum embedding size. Defaults to 600.
    Returns
    -------
    int
        embedding size
    """
    if nb_categories > 2:
        return min(round(multiplicative_factor * nb_categories**power_exponent), max_size)
    else:
        return 1


def assert_monotone(
    inputs: np.ndarray,
    outputs: np.ndarray,
    non_decreasing: Optional[bool] = None,
    error_message: str = "",
    tol: float = 1e-5,
) -> None:
    """
    Asserts if the `outputs` are a monontone function of the `inputs`.
    `inputs` should be a 1-dimensional.
    `outputs` should be a 1 or 2-dimensional array. If 2-dimensional,
    then every row is tested to be a monotone function of the inputs.
    If `non_decreasing` is `None` then an exception is raised only if the mapping
    is neither non-decreasing, nor non-increasing.
    If `non_decreasing` is `True` then an exception is raised only if the mapping
    is not non-decreasing.
    If `non_decreasing` is `False` then an exception is raised only if the mapping
    is not non-increasing.
    """
    if len(inputs.shape) != 1:
        raise ValueError("inputs should be a 1-dimensional array")

    if len(outputs.shape) not in [1, 2]:
        raise ValueError("outputs should be a 1 or 2-dimensional array")

    if inputs.shape[0] != outputs.shape[-1]:
        raise ValueError("the last dimension of outputs should match the dimension of inputs")

    argsort_indices = np.argsort(inputs)

    is_non_decreasing = np.all(np.diff(outputs[..., argsort_indices], n=1, axis=1) >= -tol)
    is_non_increasing = np.all(np.diff(outputs[..., argsort_indices], n=1, axis=1) <= tol)

    if (non_decreasing is None) and not (is_non_decreasing or is_non_increasing):
        raise NotMonotone(error_message)
    elif (non_decreasing is True) and not is_non_decreasing:
        raise NotNonIncreasing(error_message)
    elif (non_decreasing is False) and not is_non_increasing:
        raise NotNonDecreasing(error_message)
