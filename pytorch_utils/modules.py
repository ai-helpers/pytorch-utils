from __future__ import annotations

import itertools
from dataclasses import dataclass
from inspect import signature
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import pytorch_lightning as pl
import torch
import torch.nn as nn

from pytorch_utils.data_modules import (
    AugmentedBernoulliDataModule,
)
from pytorch_utils.utils import (
    BatchTorchTensors,
    CategoricalFeatureEmbedding,
    NamedTorchMetrics,
    NamedTorchTensors,
    assert_monotone,
)


class LinearNonNeg(nn.Linear):
    """
    Alternative linear layer with nonnegative weights (bias unchanged).
    This ensures the outputs are always a non-decreasing function of the inputs
    (no matter the values of parameters `self.weight` and `self.bias`, which may vary during training).

    The easiest way to implement this class with minimal code is to subclass `torch.nn.Linear` and
    apply a positive transformation (namely `torch.nn.functional.elu` shifted by 1) to the weights
    before applying the linear transformation in the `forward` method.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(
            input,
            torch.nn.functional.elu(  # this is the only difference with the original torch.nn.Linear module
                self.weight, alpha=1.0, inplace=False
            )
            + 1,
            self.bias,
        )


class BatchNorm1dNonNeg(nn.BatchNorm1d):
    """
    Alternative batch normalization with nonnegative weights (bias unchanged).
    This ensures the outputs are always a non-decreasing function of the inputs
    when `self.training=False` (no matter the values of parameters `self.weight` and `self.bias`,
    which may vary during training).

    The easiest way to implement this class with minimal code is to subclass
    `torch.nn.BatchNorm1d` and apply a positive transformation (namely `torch.nn.functional.elu` shifted by 1)
    to the weights before applying the batch norm transformation in the `forward` method.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """

        return torch.nn.functional.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            torch.nn.functional.elu(  # this is the only difference with the original torch.nn.Linear module
                self.weight, alpha=1.0, inplace=False
            )
            + 1,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class LinearSemiNonNeg(nn.Module):
    """
    Alternative linear layer combining a standard linear layer (`torch.nn.Linear`)
    together with a `LinearNonNeg` layer (by summing the two).
    The outputs are always a non-decreasing function of the inputs named `non_neg_inputs_name`
    (no matter the weights and biases).
    The outputs are not necessarily monotone w.r.t. the inputs named `other_inputs_name`.
    """

    def __init__(
        self,
        in_features_non_neg,
        in_features_others,
        out_features,
        non_neg_inputs_name="non_neg_inputs",
        other_inputs_name="other_inputs",
    ):
        super().__init__()
        self.in_features_non_neg = in_features_non_neg
        self.in_features_others = in_features_others
        self.out_features = out_features
        self.non_neg_inputs_name = non_neg_inputs_name
        self.other_inputs_name = other_inputs_name

        self.linear_non_neg = LinearNonNeg(
            in_features=self.in_features_non_neg,
            out_features=self.out_features,
            bias=False,  # duplicating biases is useless
        )
        self.linear_others = nn.Linear(
            in_features=self.in_features_others,
            out_features=self.out_features,
            bias=True,
        )

    def forward(self, input: NamedTorchTensors):
        return self.linear_non_neg(input[self.non_neg_inputs_name]) + self.linear_others(
            input[self.other_inputs_name]
        )


class BiLinearSemiNonNeg(nn.Module):
    def __init__(
        self,
        in_features_non_neg,
        in_features_others,
        out_features_non_neg,
        out_features_others,
        non_neg_inputs_name="non_neg_inputs",
        other_inputs_name="other_inputs",
    ):
        """
        Yet another custom layer that concatenates a standard
        linear layer (`torch.nn.Linear`) together with a `LinearNonNeg`
        layer (keeping the two layers separate).
        """
        super().__init__()
        self.in_features_non_neg = in_features_non_neg
        self.in_features_others = in_features_others
        self.out_features_non_neg = out_features_non_neg
        self.out_features_others = out_features_others
        self.non_neg_inputs_name = non_neg_inputs_name
        self.other_inputs_name = other_inputs_name

        self.linear_semi_non_neg = LinearSemiNonNeg(
            in_features_non_neg=self.in_features_non_neg,
            in_features_others=self.in_features_others,
            out_features=self.out_features_non_neg,
            non_neg_inputs_name=self.non_neg_inputs_name,
            other_inputs_name=self.other_inputs_name,
        )
        self.linear = nn.Linear(
            in_features=self.in_features_others,
            out_features=self.out_features_others,
            bias=True,
        )

    def forward(self, input: NamedTorchTensors):
        return {
            self.other_inputs_name: self.linear(input[self.other_inputs_name]),
            self.non_neg_inputs_name: self.linear_semi_non_neg(input),
        }


class Partitioned(nn.Module):
    """
    Unlike `torch.nn.Sequential` wich “chains” outputs to inputs
    sequentially for each module in a provided list, this module
    simultaneously transforms every partition of the input in parallel
    using the corresponding module.
    The difference between `torch.nn.Sequential` and `Partitioned` is
    similar to the difference between a series and parallel electric circuit.
    """

    def __init__(self, **module_partitions: nn.Module):
        super().__init__()
        self.modules_dict = nn.ModuleDict(module_partitions)

    def forward(self, input_partitions: NamedTorchTensors):
        return {k: module(input_partitions[k]) for k, module in self.modules_dict.items()}


class ShiftedEmbedding(nn.Embedding):
    """
    Custom embedding module that shifts all indices by 1.
    The original `torch.nn.Embedding` layer only accepts non-negative integers as inputs.
    This custom layer accepts non-negative integers and -1 as inputs.
    This is useful when -1 is used to encode unknown and/or missing values
    (i.e., using a sklearn.preprocessing.OrdinalEncoder with unknown_value=-1 and/or encoded_missing_value=-1).

    The easiest way to implement this class with minimal code is to subclass
    `torch.nn.Embedding` and shift the inputs by 1 in `forward` method.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(
            1 + input,  # this is the only difference with the original torch.nn.Embedding module
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class MeanImputationEmbedding(nn.Embedding):
    """
    Custom embedding module that applies "mean imputation" when inputs negative.

    The original `torch.nn.Embedding` layer only accepts non-negative integers as inputs.
    This custom layer also accepts negative integers as inputs.
    This is useful when for instance -1 is used to encode unknown and/or missing values
    (i.e., using a sklearn.preprocessing.OrdinalEncoder with unknown_value=-1 and/or encoded_missing_value=-1).
    When a negative input is provided, all embeddings are averaged (form of "mean imputation").
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.where(
            (input > -1)[..., None],
            nn.functional.embedding(
                nn.ReLU()(input),
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ),
            nn.functional.embedding(
                torch.tensor([0], device=input.device),
                self.weight.mean(dim=0)[None, ...],
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ),
        )


class MonotoneBernoulliProbability(pl.LightningModule):
    """
    Predicts the probability of success of an event conditioned on some features.
    The structure of the neural network enforces that the predicted probability
    is a monotone (i.e., non-decreasing and/or non-increasing) function of some specified
    features.

    `optimizer_params` should at least contain the keys `class` and `lr`

    Two ways of doing inference:
        - use `self.predict` directly
        - use method `predict` of `pytorch_lightning.Trainer`
    """

    module_scope: str

    def __init__(
        self,
        real_features_non_decreasing: List[str],
        real_features_non_increasing: List[str],
        real_features_non_monotone: List[str],
        categorical_feature_embeddings: List[CategoricalFeatureEmbedding] = [],
        hidden_sizes_monotone: List[int] = [],
        hidden_sizes_non_monotone: List[int] = [],
        polynomial_real_features_expansions: Dict[str, List[int]] = dict(),
        activation_layer_monotone: Type[torch.nn.Module] = nn.ReLU,
        activation_layer_non_monotone: Type[torch.nn.Module] = nn.ReLU,
        normalization_layer_monotone: Type[torch.nn.Module] = BatchNorm1dNonNeg,
        normalization_layer_non_monotone: Type[torch.nn.Module] = nn.BatchNorm1d,
        dropout_rate_monotone: int = 0,
        dropout_rate_non_monotone: int = 0,
        optim_criterion_params: Dict[str, Any] = {
            "class": nn.BCEWithLogitsLoss,
        },
        optimizer_params: Dict[str, Any] = {
            "class": torch.optim.Adam,
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
        },
        validation_metrics: NamedTorchMetrics = {},
        test_metrics: NamedTorchMetrics = {},
    ):
        super().__init__()

        # Check inputs
        # -------------------
        if set(real_features_non_decreasing).intersection(real_features_non_increasing):
            raise ValueError(
                "real_features_non_decreasing and real_features_non_increasing must be disjoint"
            )

        if set(real_features_non_decreasing).intersection(real_features_non_monotone):
            raise ValueError(
                "real_features_non_decreasing and real_features_non_monotone must be disjoint"
            )

        if set(real_features_non_increasing).intersection(real_features_non_monotone):
            raise ValueError(
                "real_features_non_increasing and real_features_non_monotone must be disjoint"
            )

        if not set(polynomial_real_features_expansions.keys()).issubset(
            set(real_features_non_decreasing)
            .union(real_features_non_increasing)
            .union(real_features_non_monotone)
        ):
            raise ValueError(
                """features in polynomial_real_features_expansions should belong to either
                real_features_non_decreasing or real_features_non_increasing or
                real_features_non_monotone"""
            )

        for feat, degrees in polynomial_real_features_expansions.items():
            for degree in degrees:
                if degree <= 1:
                    raise ValueError("all degrees of polynom expansions should be > 1")
                if (
                    (feat in real_features_non_decreasing) or (feat in real_features_non_increasing)
                ) and degree % 2 == 0:
                    raise ValueError(
                        """the degrees of polynom expansions should be odd integers 
                        for monotone (i.e., non-decreasing or non-increasing) features
                        so as to preserve monotonicity"""
                    )

        # Features
        # -------------------
        self.real_features_non_decreasing = real_features_non_decreasing
        self.real_features_non_increasing = real_features_non_increasing
        self.real_features_monotone = (
            self.real_features_non_decreasing + self.real_features_non_increasing
        )
        self.real_features_non_monotone = real_features_non_monotone
        self.polynomial_real_features_expansions = polynomial_real_features_expansions
        self.categorical_feature_embeddings = sorted(categorical_feature_embeddings)  # type: ignore
        self.categorical_features = [f.feature_name for f in self.categorical_feature_embeddings]
        self.size_real_features_non_decreasing = len(self.real_features_non_decreasing) + sum(
            [
                len(degrees)
                for feat, degrees in self.polynomial_real_features_expansions.items()
                if feat in self.real_features_non_decreasing
            ]
        )
        self.size_real_features_non_increasing = len(self.real_features_non_increasing) + sum(
            [
                len(degrees)
                for feat, degrees in self.polynomial_real_features_expansions.items()
                if feat in self.real_features_non_increasing
            ]
        )
        self.size_real_features_monotone = (
            self.size_real_features_non_decreasing + self.size_real_features_non_increasing
        )
        self.size_real_features_non_monotone = len(self.real_features_non_monotone) + sum(
            [
                len(degrees)
                for feat, degrees in self.polynomial_real_features_expansions.items()
                if feat in self.real_features_non_monotone
            ]
        )
        self.size_embeddings = sum(map(lambda x: x.embedding_size, categorical_feature_embeddings))
        self.size_features = (
            self.size_real_features_monotone
            + self.size_real_features_non_monotone
            + self.size_embeddings
        )
        self._monotone_feat_name = "monotone_features"
        self._non_monotone_feat_name = "non_monotone_features"

        # Neural nets
        # -------------------
        self.hidden_sizes_monotone = hidden_sizes_monotone
        self.hidden_sizes_non_monotone = hidden_sizes_non_monotone

        self.activation_layer_monotone = activation_layer_monotone
        self.activation_layer_non_monotone = activation_layer_non_monotone
        self.normalization_layer_monotone = normalization_layer_monotone
        self.normalization_layer_non_monotone = normalization_layer_non_monotone
        self.dropout_rate_monotone = dropout_rate_monotone
        self.dropout_rate_non_monotone = dropout_rate_non_monotone
        self._build_neural_nets()

        # Optimization
        # -------------------
        self.optim_criterion_params = optim_criterion_params
        self._build_optim_criterion()
        self.optimizer_params = optimizer_params

        # Metrics
        # -------------------
        # Cloning the metrics is safer: modular metrics contain internal states that
        # should belong to only one DataLoader, it is recommended to initialize a separate modular metric instances
        # for each DataLoader and in particular use separate metrics for training, validation and testing
        # see: https://torchmetrics.readthedocs.io/en/latest/pages/lightning.html?highlight=
        # modular%20metrics%20contain%20internal%20states#common-pitfalls
        self.validation_metrics = self._clone_metrics(validation_metrics)
        self.test_metrics = self._clone_metrics(test_metrics)

        # Save hyper-parameters
        # -------------------
        self.save_hyperparameters()  # saves all constructor params by default

    @property
    def learning_rate(self) -> float:
        return self.optimizer_params["lr"]

    @learning_rate.setter
    def learning_rate(self, learning_rate) -> None:
        """
        Updates learning rate (useful to apply `pytorch_lightning.trainer.Trainer.tune.lr_find`)
        """
        self.optimizer_params.update({"lr": learning_rate})

    def forward(self, x: NamedTorchTensors) -> torch.Tensor:
        return self._forward_from_logits(self._logits(x))

    def _forward_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return self.final_sigmoid_layer(logits)

    def configure_optimizers(self):
        optimizer_class = self.optimizer_params.pop("class")
        try:
            optimizer = optimizer_class(self.parameters(), **self.optimizer_params)
        finally:
            self.optimizer_params["class"] = optimizer_class
        return optimizer

    @staticmethod
    def _clone_metrics(metrics: NamedTorchMetrics) -> nn.ModuleDict:
        return nn.ModuleDict(
            {metric_name: metric.clone() for metric_name, metric in metrics.items()}
        )

    def _shared_step(
        self,
        batch: BatchTorchTensors,
        batch_idx: int,
    ) -> Tuple[NamedTorchTensors, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        features, targets, weights = batch
        batch_size = len(targets)
        logits = self._logits(features)
        preds = self._forward_from_logits(logits)
        loss = torch.mean(
            self.optim_criterion(logits, targets.float()) * (weights if weights is not None else 1)
        )
        return features, targets, weights, preds, loss, batch_size  # type: ignore

    def _evaluate_metrics(
        self,
        metrics: NamedTorchMetrics,
        preds: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        for metric in metrics.values():
            if weights is None or len(signature(metric.update).parameters) < 4:
                metric(preds, targets)
            else:
                metric(preds, targets, weights)

    def training_step(self, batch: BatchTorchTensors, batch_idx: int) -> torch.Tensor:
        _, _, _, _, loss, batch_size = self._shared_step(batch, batch_idx)  # type: ignore
        self.log(
            "training_loss",
            loss,  # type: ignore
            batch_size=batch_size,  # type: ignore
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss  # type: ignore

    def validation_step(self, batch: BatchTorchTensors, batch_idx: int) -> torch.Tensor:
        _, targets, weights, preds, loss, batch_size = self._shared_step(batch, batch_idx)  # type: ignore
        self.log(
            "validation_loss",
            loss,  # type: ignore
            batch_size=batch_size,  # type: ignore
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        if self.validation_metrics:
            self._evaluate_metrics(self.validation_metrics, preds, targets, weights)  # type: ignore
            self.log_dict(
                self.validation_metrics,  # type: ignore
                batch_size=batch_size,  # type: ignore
                on_step=False,
                on_epoch=True,
            )
        return loss  # type: ignore

    def test_step(self, batch: BatchTorchTensors, batch_idx: int) -> torch.Tensor:
        _, targets, weights, preds, loss, batch_size = self._shared_step(batch, batch_idx)  # type: ignore
        self.log(
            "test_loss",
            loss,  # type: ignore
            batch_size=batch_size,  # type: ignore
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        if self.test_metrics:
            self._evaluate_metrics(self.test_metrics, preds, targets, weights)  # type: ignore
            self.log_dict(
                self.test_metrics,  # type: ignore
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )
        return loss  # type: ignore

    def predict_step(
        self, batch: NamedTorchTensors, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        return self(batch)

    def _to_self_device(self, x: NamedTorchTensors):
        return {feat: x[feat] for feat in x}  # type: ignore

    def predict(
        self,
        features: NamedTorchTensors,
    ) -> torch.Tensor:
        training = self.training
        try:
            self.train(False)
            with torch.no_grad():
                predictions = self.predict_step(
                    self._to_self_device(features),
                    batch_idx=0,
                )
        finally:
            self.train(training)
        return predictions

    def predict_from_pandas(
        self,
        features: pd.DataFrame,
        data_module: AugmentedBernoulliDataModule,
        scaling_factors: np.ndarray = np.array([1.0]),
        min_augmented_value: float = -float("inf"),
        max_augmented_value: float = float("inf"),
    ) -> torch.Tensor:
        return self.predict(
            data_module.augment_transform_to_tensors(  # type: ignore
                features,
                scaling_factors,
                min_augmented_value,
                max_augmented_value,
            )
        )

    def _polynom_expansion(self, x: NamedTorchTensors) -> List[torch.Tensor]:
        return list(
            itertools.chain(
                *[
                    [x[feat] ** degree for degree in degrees]
                    for feat, degrees in self.polynomial_real_features_expansions.items()
                    if feat in x
                ]
            )
        )

    def _logits(self, x: NamedTorchTensors) -> torch.Tensor:
        # Encode categorical features into embeddings
        # -------------------
        categorical_embeddings = (
            [torch.cat([t for t in self.categorical_embeddings_net(x).values()], dim=1)]
            if len(self.categorical_feature_embeddings) > 0
            else []
        )

        # Polynom expansions of real features
        # -------------------
        non_monotone_polynom_expansions = self._polynom_expansion(
            {feat: x[feat] for feat in x if feat in self.real_features_non_monotone}
        )
        monotone_polynom_expansions = self._polynom_expansion(
            {
                feat: (x[feat] if feat in self.real_features_non_decreasing else torch.neg(x[feat]))
                for feat in x
                if feat in self.real_features_monotone
            }
        )

        # Apply core layers
        # -------------------
        non_monotone_features = torch.cat(
            categorical_embeddings
            + [x[feat] for feat in self.real_features_non_monotone]
            + non_monotone_polynom_expansions,
            dim=1,
        )
        monotone_features = torch.cat(
            [x[feat] for feat in self.real_features_non_decreasing]
            + [torch.neg(x[feat]) for feat in self.real_features_non_increasing]
            + monotone_polynom_expansions,
            dim=1,
        )

        intermediate_outputs = self.core_sequential_layers(
            {
                self._monotone_feat_name: monotone_features,
                self._non_monotone_feat_name: non_monotone_features,
            }
        )

        # Apply missing layers
        # -------------------
        logits = self.final_linear_layer(
            {
                self._monotone_feat_name: self.missing_sequential_layers_monotone(
                    intermediate_outputs[self._monotone_feat_name]
                ),
                self._non_monotone_feat_name: self.missing_sequential_layers_non_monotone(
                    intermediate_outputs[self._non_monotone_feat_name]
                ),
            }
        )

        return logits.view(-1)

    def _build_optim_criterion(self) -> None:
        optim_criterion_class = self.optim_criterion_params.pop("class")
        try:
            self.optim_criterion = optim_criterion_class(**self.optim_criterion_params)
        finally:
            self.optim_criterion_params["class"] = optim_criterion_class

    def _build_neural_nets(self) -> None:
        # Embeddings for categorical features
        # -------------------
        self._build_categorical_embeddings()

        # Core sequential layers
        # -------------------
        self._build_core_sequential_layers()

        # Missing layers
        # -------------------
        self._build_missing_layers()

        # Final layer
        # -------------------
        self._build_final_layer()

    def _build_categorical_embeddings(self) -> None:
        self.categorical_embeddings_net = Partitioned(
            **{
                cat_feat_emb.feature_name: MeanImputationEmbedding(  # ShiftedEmbedding(
                    num_embeddings=cat_feat_emb.nb_distinct_values,
                    embedding_dim=cat_feat_emb.embedding_size,
                )
                for cat_feat_emb in self.categorical_feature_embeddings
            }
        )

    def _build_core_sequential_layers(self) -> None:
        self.core_sequential_layers = nn.Sequential(
            *[
                nn.Sequential(
                    BiLinearSemiNonNeg(
                        in_features_non_neg=in_monotone,
                        in_features_others=in_non_monotone,
                        out_features_non_neg=out_monotone,
                        out_features_others=out_non_monotone,
                        non_neg_inputs_name=self._monotone_feat_name,
                        other_inputs_name=self._non_monotone_feat_name,
                    ),
                    Partitioned(
                        **{
                            self._monotone_feat_name: self.activation_layer_monotone(),
                            self._non_monotone_feat_name: self.activation_layer_non_monotone(),
                        }
                    ),
                    Partitioned(
                        **{
                            self._monotone_feat_name: nn.Dropout(p=self.dropout_rate_monotone),
                            self._non_monotone_feat_name: nn.Dropout(
                                p=self.dropout_rate_non_monotone
                            ),
                        }
                    ),
                    Partitioned(
                        **{
                            self._monotone_feat_name: self.normalization_layer_monotone(
                                out_monotone
                            ),
                            self._non_monotone_feat_name: self.normalization_layer_non_monotone(
                                out_non_monotone
                            ),
                        }
                    ),
                )
                for in_non_monotone, out_non_monotone, in_monotone, out_monotone in zip(
                    [self.size_real_features_non_monotone + self.size_embeddings]
                    + self.hidden_sizes_non_monotone[:-1],
                    self.hidden_sizes_non_monotone,
                    [self.size_real_features_monotone] + self.hidden_sizes_monotone[:-1],
                    self.hidden_sizes_monotone,
                )
            ]
        )

    def _build_missing_layers(self) -> None:
        # Monotone layers
        nb_missing_layers_monotone = max(
            len(self.hidden_sizes_monotone) - len(self.core_sequential_layers), 0
        )
        self.missing_sequential_layers_monotone = nn.Sequential(
            *[
                nn.Sequential(
                    LinearNonNeg(in_features=in_features, out_features=out_features),
                    self.activation_layer_monotone(),
                    nn.Dropout(p=self.dropout_rate_monotone),
                    self.normalization_layer_monotone(out_features),
                )
                for in_features, out_features in zip(
                    self.hidden_sizes_monotone[-nb_missing_layers_monotone - 1 : -1],
                    self.hidden_sizes_monotone[-nb_missing_layers_monotone:],
                )
            ]
        )

        # Non_monotone layers
        nb_missing_layers_non_monotone = max(
            len(self.hidden_sizes_non_monotone) - len(self.core_sequential_layers), 0
        )
        self.missing_sequential_layers_non_monotone = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=out_features),
                    self.activation_layer_non_monotone(),
                    nn.Dropout(p=self.dropout_rate_non_monotone),
                    self.normalization_layer_non_monotone(out_features),
                )
                for in_features, out_features in zip(
                    self.hidden_sizes_non_monotone[-nb_missing_layers_non_monotone - 1 : -1],
                    self.hidden_sizes_non_monotone[-nb_missing_layers_non_monotone:],
                )
            ]
        )

    def _build_final_layer(self) -> None:
        self.final_linear_layer = LinearSemiNonNeg(
            in_features_non_neg=(
                self.hidden_sizes_monotone[-1]
                if len(self.hidden_sizes_monotone) > 0
                else self.size_real_features_monotone
            ),
            in_features_others=(
                self.hidden_sizes_non_monotone[-1]
                if len(self.hidden_sizes_non_monotone) > 0
                else self.size_real_features_non_monotone + self.size_embeddings
            ),
            out_features=1,
            non_neg_inputs_name=self._monotone_feat_name,
            other_inputs_name=self._non_monotone_feat_name,
        )
        self.final_sigmoid_layer = nn.Sigmoid()

    def probability_mapping(
        self,
        data_module: AugmentedBernoulliDataModule,
        other_features: pd.DataFrame,
        min_value: float,
        max_value: float,
        nb_points: int = 100,
        trainer: Optional[pl.Trainer] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Computes the mapping between covariate `data_module.augmented_col` and the predicted probability
        of the neural network on the closed interval `[min_value, max_value]`,
        all `other_features` being fixed.
        The mapping is discretized into `nb_points` points.

        There are two ways to use this function:
        - call with `trainer=None` => method `self.predict` is used directly for
        inference
        - call with `trainer=pytorch_lightning.Trainer(...)` => method `predict` of
        `pytorch_lightning.Trainer` is used for inference
        """
        other_features = other_features.copy()  # make a copy before making modifications
        nb_samples = len(other_features)
        real_feature_values = np.array(
            [min_value + (max_value - min_value) * i / (nb_points - 1) for i in range(nb_points)],
            dtype=np.float32,
        )
        other_features[data_module.augmented_col] = np.array(1, dtype=np.float32)

        if trainer:
            data_module.prediction_df = other_features
            data_module.prediction_scaling_factors = real_feature_values
            self.prediction_min_augmented_value = min_value
            self.prediction_max_augmented_value = max_value
            bb_conversion_proba = torch.cat(trainer.predict(self, data_module))  # type: ignore
        else:
            bb_conversion_proba = self.predict_from_pandas(
                features=other_features,
                data_module=data_module,
                scaling_factors=real_feature_values,
                min_augmented_value=min_value,
                max_augmented_value=max_value,
            )
        other_features.drop(columns=[data_module.augmented_col], inplace=True)

        return (
            other_features,
            real_feature_values,
            torch.transpose(bb_conversion_proba.view(nb_points, nb_samples), dim0=0, dim1=1),
        )

    def assert_monotone_probability(
        self,
        data_module: AugmentedBernoulliDataModule,
        other_features: pd.DataFrame,
        non_decreasing: bool,
        min_value: float,
        max_value: float,
        nb_points: int = 100,
        trainer: Optional[pl.Trainer] = None,
        error_message: str = "",
    ):
        (
            other_features,
            real_feature_values,
            bb_conversion_proba,
        ) = self.probability_mapping(
            data_module=data_module,
            other_features=other_features,
            min_value=min_value,
            max_value=max_value,
            nb_points=nb_points,
            trainer=trainer,
        )

        assert_monotone(
            inputs=real_feature_values,
            outputs=bb_conversion_proba.numpy(force=True),
            non_decreasing=non_decreasing,
            error_message=error_message,
        )

    def plot_probability_mapping(
        self,
        data_module: AugmentedBernoulliDataModule,
        other_features: pd.DataFrame,
        min_value: float,
        max_value: float,
        nb_points: int = 100,
        x_title: str = "Covariate",
        y_title: str = "Predicted probability",
        title: str = "Evolution of the predicted probability as a function of the covariate",
        trainer: Optional[pl.Trainer] = None,
    ) -> plotly.graph_objects.Figure:
        """
        Plots the mapping between covariate `data_module.augmented_col` and the predicted probability
        of the neural network on the closed interval `[min_value, max_value]`,
        all `other_features` being fixed.
        The mapping is discretized into `nb_points` points.

        There are two ways to use this function:
        - call with `trainer=None` => method `self.predict` is used directly for
        inference
        - call with `trainer=pytorch_lightning.Trainer(...)` => method `predict` of
        `pytorch_lightning.Trainer` is used for inference
        """
        (
            other_features,
            real_feature_values,
            bb_conversion_proba,
        ) = self.probability_mapping(
            data_module=data_module,
            other_features=other_features,
            min_value=min_value,
            max_value=max_value,
            nb_points=nb_points,
            trainer=trainer,
        )

        nb_samples = len(other_features)
        informative_features = [
            f for f in data_module.output_features if f not in [data_module.augmented_col]
        ]
        other_features.index.names = ["Sample"]
        df = other_features.loc[
            np.repeat(other_features.index, nb_points), informative_features
        ].reset_index(drop=False)  # , names="Sample"
        df[x_title] = list(real_feature_values) * nb_samples
        df[y_title] = bb_conversion_proba.reshape(-1).numpy(force=True)

        fig = px.line(
            df,
            x=x_title,
            y=y_title,
            color="Sample",
            title=title,
            hover_data=informative_features,
        )
        fig.update_layout(hovermode="closest")
        return fig


Module = TypeVar("Module", bound=MonotoneBernoulliProbability)
DataModule = TypeVar("DataModule", bound=AugmentedBernoulliDataModule)


@dataclass(frozen=True)
class ProbabilityPredictor(Generic[Module, DataModule]):
    """
    Just a pair `(MonotoneBernoulliProbability, AugmentedBernoulliDataModule)`
    with useful methods such as `predict_from_pandas`.
    """

    module: Module
    data_module: DataModule

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        module_class: Type[Module] = MonotoneBernoulliProbability,
        data_module_class: Type[DataModule] = AugmentedBernoulliDataModule,  # type: ignore
        clear_data: bool = False,  # allows to save memory (clears training/validation/test data)
        compile_module: bool = False,  # allows to speed up inference on GPU
        compilation_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> ProbabilityPredictor:
        data_module = data_module_class.load_from_checkpoint(checkpoint_path, **kwargs)
        module = module_class.load_from_checkpoint(checkpoint_path, **kwargs)
        proba_predictor = cls(
            module=torch.compile(module, **compilation_kwargs) if compile_module else module,
            data_module=data_module,
        )
        return proba_predictor.clear_data() if clear_data else proba_predictor

    def clear_data(self) -> ProbabilityPredictor[Module, DataModule]:
        self.data_module.clear_data()
        return self

    def predict_from_pandas(
        self,
        context: pd.DataFrame,
    ) -> np.ndarray:
        return self.module.predict(
            self.data_module.transform_to_tensors(context)  # type: ignore
        ).numpy(force=True)
