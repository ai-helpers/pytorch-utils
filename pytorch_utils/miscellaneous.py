from typing import Hashable, List, TypeVar, Protocol, ClassVar, Any, Dict


def class_full_name(cls):
    return ".".join([cls.__module__, cls.__name__])


T = TypeVar("T", bound=Hashable)


def sort_hashable(array: List[T]) -> List[T]:
    return sorted(array, key=lambda x: hash(x))


class DataclassType(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Any]]
