from typing import Any, Literal, Optional, Tuple, Union

import torch
from sklearn.calibration import CalibrationDisplay
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import BinaryCalibrationError
from torchmetrics.utilities.data import dim_zero_cat


class WeightedMeanSquaredError(MeanSquaredError):
    """
    Analogue of `torchmetrics.MeanSquaredError` but with (optional) sample weights.
    """

    def __init__(
        self,
        squared: bool = True,
        **kwargs: Any,
    ) -> None:
        super(MeanSquaredError, self).__init__(**kwargs)

        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.squared = squared

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> None:  # type: ignore
        """Update state with predictions and targets."""
        preds = preds if preds.is_floating_point is not None else preds.float()
        target = target if target.is_floating_point is not None else target.float()
        diff = preds - target
        sum_squared_error = torch.sum(
            diff * diff * (sample_weights if sample_weights is not None else 1)
        )
        self.sum_squared_error += sum_squared_error
        self.total += torch.sum(sample_weights) if sample_weights is not None else target.numel()


class WeightedMeanAbsoluteError(MeanAbsoluteError):
    """
    Analogue of `torchmetrics.MeanAbsoluteError` but with (optional) sample weights.
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> None:  # type: ignore
        """Update state with predictions and targets."""
        preds = preds if preds.is_floating_point is not None else preds.float()
        target = target if target.is_floating_point is not None else target.float()
        diff = preds - target
        sum_abs_error = torch.sum(
            torch.abs(diff) * (sample_weights if sample_weights is not None else 1)
        )
        self.sum_abs_error += sum_abs_error
        self.total += torch.sum(sample_weights) if sample_weights is not None else target.numel()


class WeightedBinaryCalibrationError(BinaryCalibrationError):
    """
    Analogue of `torchmetrics.classification import BinaryCalibrationError` but with (optional) sample weights.
    """

    def __init__(
        self,
        n_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_bins=n_bins,
            norm=norm,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs,
        )
        self.add_state("weights", [], dist_reduce_fx="cat")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> None:  # type: ignore
        super().update(preds=preds, target=target)
        if weights is not None:
            self.weights.append(weights)  # type: ignore

    def compute(self) -> torch.Tensor:
        confidences = dim_zero_cat(self.confidences)  # type: ignore
        accuracies = dim_zero_cat(self.accuracies)  # type: ignore
        weights = dim_zero_cat(self.weights) if self.weights else None  # type: ignore

        return self._ce_compute(
            confidences=confidences,
            accuracies=accuracies,
            weights=weights,
            bin_boundaries=self.n_bins,
            norm=self.norm,
        )

    def _ce_compute(
        self,
        confidences: torch.Tensor,
        accuracies: torch.Tensor,
        weights: Optional[torch.Tensor],
        bin_boundaries: Union[torch.Tensor, int],
        norm: str = "l1",
        debias: bool = False,
    ) -> torch.Tensor:
        """
        Analogue of `torchmetrics.functional.classification.calibration_error._ce_compute` but with weights.
        Computes the calibration error given the provided bin boundaries and norm.
        Args:
            confidences: The confidence (i.e. predicted prob) of the top1 prediction.
            accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
            bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.
            norm: Norm function to use when computing calibration error. Defaults to "l1".
            debias: Apply debiasing to L2 norm computation as in
                `Verified Uncertainty Calibration`_. Defaults to False.
        Raises:
            ValueError: If an unsupported norm function is provided.
        Returns:
            Tensor: Calibration error scalar.
        """
        if isinstance(bin_boundaries, int):
            bin_boundaries = torch.linspace(
                0, 1, bin_boundaries + 1, dtype=torch.float, device=confidences.device
            )

        if norm not in {"l1", "l2", "max"}:
            raise ValueError(f"Norm {norm} is not supported. Please select from l1, l2, or max. ")  # nosec B608

        with torch.no_grad():
            acc_bin, conf_bin, prop_bin = self._binning_bucketize(
                confidences, accuracies, weights, bin_boundaries
            )

        if norm == "l1":
            ce = torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
        elif norm == "max":
            ce = torch.max(torch.abs(acc_bin - conf_bin))
        elif norm == "l2":
            ce = torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
            # NOTE: debiasing is disabled in the wrapper functions. This implementation differs from that in sklearn.
            if debias:
                # the order here (acc_bin - 1 ) vs (1 - acc_bin) is flipped from
                # the equation in Verified Uncertainty Prediction (Kumar et al 2019)/
                debias_bins = (acc_bin * (acc_bin - 1) * prop_bin) / (
                    prop_bin * accuracies.size()[0] - 1
                )
                ce += torch.sum(
                    torch.nan_to_num(debias_bins)
                )  # replace nans with zeros if nothing appeared in a bin
            ce = torch.sqrt(ce) if ce > 0 else torch.tensor(0)
        return ce

    def _binning_bucketize(
        self,
        confidences: torch.Tensor,
        accuracies: torch.Tensor,
        weights: Optional[torch.Tensor],
        bin_boundaries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Analogue of `torchmetrics.functional.classification.calibration_error._binning_bucketize` but with weights.
        Compute calibration bins using ``torch.bucketize``. Use for pytorch >= 1.6.
        Args:
            confidences: The confidence (i.e. predicted prob) of the top1 prediction.
            accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
            bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.
        Returns:
            tuple with binned accuracy, binned confidence and binned probabilities
        """
        accuracies = accuracies.to(dtype=confidences.dtype)
        acc_bin = torch.zeros(
            len(bin_boundaries) - 1, device=confidences.device, dtype=confidences.dtype
        )
        conf_bin = torch.zeros(
            len(bin_boundaries) - 1, device=confidences.device, dtype=confidences.dtype
        )
        # count_bin = torch.zeros(len(bin_boundaries) - 1, device=confidences.device, dtype=confidences.dtype)
        weights_bin = torch.zeros(
            len(bin_boundaries) - 1,
            device=confidences.device,
            dtype=confidences.dtype if weights is None else weights.dtype,
        )

        indices = torch.clamp(
            torch.bucketize(confidences, bin_boundaries, right=True) - 1,
            min=0,
            max=len(weights_bin) - 1,
        )

        # count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))
        weights_bin.scatter_add_(
            dim=0,
            index=indices,
            src=torch.ones_like(confidences) if weights is None else weights,
        )

        conf_bin.scatter_add_(
            dim=0, index=indices, src=confidences * (1 if weights is None else weights)
        )
        # conf_bin = torch.nan_to_num(conf_bin / count_bin)
        conf_bin = torch.nan_to_num(conf_bin / weights_bin)

        acc_bin.scatter_add_(
            dim=0, index=indices, src=accuracies * (1 if weights is None else weights)
        )
        # acc_bin = torch.nan_to_num(acc_bin / count_bin)
        acc_bin = torch.nan_to_num(acc_bin / weights_bin)

        # prop_bin = count_bin / count_bin.sum()
        prop_bin = weights_bin / weights_bin.sum()

        # Backup acc_bin, conf_bin, prop_bin to be used in calibration_curve method
        self.acc_bin_backup = acc_bin
        self.conf_bin_backup = conf_bin
        self.prop_bin = prop_bin

        return acc_bin, conf_bin, prop_bin

    def calibration_curve(self, estimator_name: str, pos_label: int = 1) -> CalibrationDisplay:
        return (
            CalibrationDisplay(
                prob_true=self.acc_bin_backup[self.prop_bin > 0].numpy(force=True),
                prob_pred=self.conf_bin_backup[self.prop_bin > 0].numpy(force=True),
                y_prob=None,
                estimator_name=estimator_name,
                pos_label=pos_label,
            )
            .plot()
            .figure_
        )
