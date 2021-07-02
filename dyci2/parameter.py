from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Union, Callable, List


class Range(ABC):
    @abstractmethod
    def __contains__(self, item: Any) -> bool:
        pass


class NominalRange(Range):
    def __init__(self, labels: Iterable[Any]):
        self.labels: Iterable[Any] = labels

    def __contains__(self, item):
        return item in self.labels


class OrdinalRange(Range):
    def __init__(self, lower_bound: Optional[Union[float, int]] = None,
                 upper_bound: Optional[Union[float, int]] = None):
        self.lower_bound: Optional[Union[float, int]] = lower_bound
        self.upper_bound: Optional[Union[float, int]] = upper_bound

    def __contains__(self, item: Any) -> bool:
        return self.lower_bound <= item <= self.upper_bound


class Parameter:
    def __init__(self, initial_value: Any, value_range: Optional[Range] = None,
                 setter: Optional[Callable[[Any], None]] = None):
        self.value = initial_value
        self.value_range = value_range
        self.setter = setter

    def set(self, value: Any) -> None:
        """ raises: ValueError if value is outside of the defined range"""
        if self.value_range is not None:
            if value in self.value_range:
                self._set(value)
            else:
                raise ValueError(f"Value {value} is outside defined range")

    def _set(self, value: Any) -> None:
        if self.setter is not None:
            self.setter(value)
        else:
            self.value = value

    def get(self) -> Any:
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
