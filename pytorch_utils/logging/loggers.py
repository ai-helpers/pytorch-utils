import logging
import os
from functools import wraps
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import (
    Any,
    Callable,
    List,
    Dict,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)
from typing_extensions import Self

import mlflow
import pandas as pd

from pytorch_utils.miscellaneous import (
    class_full_name,
    DataclassType,
)
from .utils import get_active_mlflow_experiment_id, get_active_mlflow_run_id


LOGGING_DISABLED = False  # Useful to disable logging globally


def disable_logging():
    global LOGGING_DISABLED
    LOGGING_DISABLED = True


def enable_logging():
    global LOGGING_DISABLED
    LOGGING_DISABLED = False


def is_logging_disabled():
    global LOGGING_DISABLED
    return LOGGING_DISABLED


@runtime_checkable
class Logger(Protocol):
    def log_param(self, key: str, value: Any) -> Any:
        return None

    def log_params(self, params: Dict[str, Any]) -> None:
        return None

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        return None

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        return None

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        return None

    def log_pandas_artifact(self, df: Union[pd.DataFrame, pd.Series], df_name: str) -> None:
        return None

    def set_tag(self, key: str, value: Any) -> None:
        return None

    def set_tags(self, tags: Dict[str, Any]) -> None:
        return None

    def sklearn_autolog(self, **kwargs) -> None:
        return None

    def pytorch_autolog(self, **kwargs) -> None:
        return None

    def autolog(self, **kwargs) -> None:
        return None

    @contextmanager
    def context(self) -> Any:
        yield None


class VoidLogger(Logger):
    """
    Does not log anything
    """


@dataclass
class PythonLogger(Logger):
    """
    Use `logging.basicConfig(level=..., filename=..., force=True)` to set the root logger level, filename, etc...
    See: https://docs.python.org/3/library/logging.html#logging.basicConfig
    """

    logger: logging.Logger = logging.getLogger()
    logging_level: int = logging.INFO

    def log_param(self, key: str, value: Any) -> None:
        self.logger.log(self.logging_level, f"Parameter {key}={value}")

    def log_params(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        self.logger.log(self.logging_level, f"Metric {key}={value}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            self.log_param(k, v)

    def set_tag(self, key: str, value: Any) -> None:
        self.logger = self.logger.getChild(value)

    def set_tags(self, tags: Dict[str, Any]) -> None:
        for k, v in tags.items():
            self.set_tag(k, v)


class MLFlowLogger(Logger):
    def __init__(self, **mlflow_run_params) -> None:
        self.mlflow_run_params = mlflow_run_params

    def log_param(self, key: str, value: Any) -> None:
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        mlflow.log_metric(key, value, step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        mlflow.log_metrics(metrics, step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        mlflow.log_artifact(local_path, artifact_path)

    def log_pandas_artifact(
        self, df: pd.DataFrame, df_name: str, artifact_path: Optional[str] = None
    ) -> None:
        tmp_path_csv = f"/TMPDIR/{df_name}.csv"
        df.reset_index().to_csv(tmp_path_csv, index=False)
        self.log_artifact(tmp_path_csv, artifact_path=artifact_path)
        os.remove(tmp_path_csv)

    def set_tag(self, key: str, value: Any) -> None:
        mlflow.set_tag(key, value)

    def set_tags(self, tags: Dict[str, Any]) -> None:
        mlflow.set_tags(tags)

    def sklearn_autolog(self, **kwargs) -> None:
        mlflow.sklearn.autolog(**kwargs)

    def pytorch_autolog(self, **kwargs) -> None:
        mlflow.pytorch.autolog(**kwargs)

    def autolog(self, **kwargs) -> None:
        mlflow.autolog(**kwargs)

    @contextmanager
    def context(self):
        with mlflow.start_run(**self.mlflow_run_params) as mlflow_run:
            yield mlflow_run


@dataclass
class ActiveRunMLFlowLogger(MLFlowLogger):
    nested: bool = False
    tags: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

    @property
    def mlflow_run_params(self):
        run_params = asdict(self)
        run_params.update({"run_id": get_active_mlflow_run_id()})
        return run_params


@dataclass
class ActiveExperimentMLFlowLogger(MLFlowLogger):
    run_name: Optional[str] = None
    nested: bool = False
    tags: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

    @property
    def mlflow_run_params(self):
        run_params = asdict(self)
        run_params.update({"experiment_id": get_active_mlflow_experiment_id()})
        return run_params


class Loggable(Protocol):
    loggers: Dict[str, Logger]


def use_loggers(*logger_names: str):
    def decorator(log_method: Callable):
        @wraps(log_method)
        def wrapper(self: Loggable, *args, **kwargs):
            if is_logging_disabled():
                log_method(self, logger=VoidLogger(), *args, **kwargs)
            else:
                for loger_name in logger_names:
                    logger = self.loggers[loger_name]
                    with logger.context():
                        log_method(self, logger=logger, *args, **kwargs)

        return wrapper

    return decorator


@runtime_checkable
class SingleLoggerDataclassLoggable(DataclassType, Protocol):
    """A specific ``Loggable``"""

    logger: Logger

    def __post_init__(self):
        self.set_logger(self.logger)

    def set_logger(self, logger: Logger) -> Self:
        object.__setattr__(self, "logger", logger)
        object.__setattr__(self, "loggers", {"logger": logger})
        return self

    @use_loggers("logger")
    def log(self, logger: Logger, params_not_to_log: List[str] = ["logger"]) -> None:
        params = asdict(self)
        params = {
            param_name: getattr(self, param_name) for param_name in params
        }  # by default, asdict explodes nested dataclasses as dict too

        for param in params.values():
            if isinstance(param, SingleLoggerDataclassLoggable):
                param.log(params_not_to_log=params_not_to_log)

        for param_name in params_not_to_log:
            del params[param_name]

        for param_name, param_value in params.items():
            if isinstance(param_value, pd.DataFrame):
                logger.log_pandas_artifact(
                    param_value,
                    df_name=param_name,
                )
                del params[param_name]

        logger.log_params(params)
        logger.log_param("type", class_full_name(self.__class__))
        logger.log_param("repr", repr(self))
