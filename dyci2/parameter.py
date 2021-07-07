from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Callable, List, TypeVar, Generic

T = TypeVar('T')


class Range(Generic[T], ABC):
    @abstractmethod
    def __contains__(self, item: Any) -> bool:
        pass


class NominalRange(Range):
    def __init__(self, labels: Iterable[T]):
        self.labels: Iterable[T] = labels

    def __contains__(self, item):
        return item in self.labels


class OrdinalRange(Range):
    def __init__(self, lower_bound: Optional[T] = None, upper_bound: Optional[T] = None):
        self.lower_bound: Optional[T] = lower_bound
        self.upper_bound: Optional[T] = upper_bound

    def __contains__(self, item: T) -> bool:
        if self.lower_bound is not None and item < self.lower_bound:
            return False
        if self.upper_bound is not None and item > self.upper_bound:
            return False
        return True


class Parameter(Generic[T]):
    def __init__(self, initial_value: T, value_range: Optional[Range] = None,
                 func: Optional[Callable[[Any], T]] = None):
        self.value: T = initial_value
        self.value_range: Optional[Range] = value_range
        self.func: Optional[Callable[[Any], T]] = func

    def set(self, value: Any) -> None:
        """ raises: ValueError if value is outside of the defined range"""
        if self.func is not None:
            self._set(self.func(value))
        else:
            self._set(value)

    def _set(self, value: Any) -> None:
        """ raises: ValueError if value is outside of the defined range"""
        if self.value_range is not None and value not in self.value_range:
            raise ValueError(f"Value {value} is outside defined range")
        else:
            self.value = value

    def get(self) -> T:
        return self.value


class Parametric:

    def set_parameter(self, parameter_path: List[str], value: Any) -> None:
        """ raises: ValueError if value is outside of the defined parameter's range,
                    KeyError if the parameter path is invalid """
        try:
            obj: Any = self.__dict__[parameter_path.pop(0)]
            if len(parameter_path) == 0:
                if isinstance(obj, Parameter):
                    obj.set(value)
                else:
                    raise KeyError("The value at the given address is not a parameter")
            elif isinstance(obj, Parametric):
                obj.set_parameter(parameter_path, value)
            else:
                raise KeyError(f"The object {str(obj)} is not part of the Parametric hierarchy")

        except IndexError:
            # case: parameter_path is empty on initialization
            raise KeyError("No valid path was provided")
