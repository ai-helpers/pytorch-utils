class MLError(Exception):
    """
    Exception class from which every exception in this library will derive.
    It enables other projects using this library to catch all errors coming
    from the library with a single "except" statement
    """

    pass


class CatalogError(MLError):
    """
    Raised when the catalog is not valid
    """

    pass


class InvalidDataFormatError(MLError):
    """
    Raised when the catalog is not valid
    """

    pass


class LocalDirNotWriteableException(MLError):
    """
    Raised when the local directory is not writeable
    """

    pass


class MissingConfigFileException(MLError):
    """
    Raised when a given configuration file cannot be found
    """

    pass


class BadCLIParameterException(MLError):
    """
    Raised when there is an issue with a parameter
    in the CLI (commnand-line interface)
    """

    pass


class BadConfigException(MLError):
    """
    Raised when a configuration file cannot be loaded, for instance
    due to wrong syntax or poor formatting.
    """

    pass


class BadConfigLogLevelException(BadConfigException):
    """
    Raised when the log level (`log_level`) does not exist
    """

    pass


class BadConfigSparkMasterException(BadConfigException):
    """
    Raised when the Spark master (`spark_master`) is not valid
    """

    pass


class BadConfigPathException(BadConfigException):
    """
    Raised when the configuration path parameter (`conf_path`) does not
    contain at least a directory/path where to find configuration files
    """

    pass


class BadConfigMissingInputException(BadConfigException):
    """
    Raised when the 'input_data'/{'products', 'transactions'} file-path
    is missing
    """

    pass


class BadConfigMissingOutputException(BadConfigException):
    """
    Raised when the 'output_data'/'transactions' file-path is missing
    """

    pass


class TaskNotFoundError(MLError):
    """Raised when task name is not found in entrypoints"""


class MissingDatasetError(MLError):
    """Raised when a dataset is not found in the catalog"""

    pass


class DataSetError(MLError):
    """Raised when there is an issue with a dataset"""

    pass


class MissingConfigException(MLError):
    """Raised when a configuration is missing"""

    pass


class BadConfigFormatException(MLError):
    """Raised when a configuration is not formatted correctly"""

    pass


class InconsistentDatasetConfigurations(MLError):
    """
    Raised when several inconsistent dataset configurations are being used
    to build a dataset.
    """

    pass


class NotMonotone(MLError):
    """
    Raised when some outputs are not a monotone function of some inputs.
    """

    pass


class NotNonDecreasing(MLError):
    """
    Raised when some outputs are not an non-decreasing function of some inputs.
    """

    pass


class NotNonIncreasing(MLError):
    """
    Raised when some outputs are not an non-increasing function of some inputs.
    """

    pass
