from abc import ABC, abstractmethod
from typing import Any, List, Union

from label import Label


class MemoryEvent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def value(self) -> Any:
        """ TODO """

    @abstractmethod
    def label(self) -> Label:
        """ TODO """

    @abstractmethod
    def max_representation(self) -> Any:
        """ TODO """


class BasicEvent(MemoryEvent):
    def __init__(self, val: Union[int, float, str], label: Label):
        super().__init__()
        self._event: Union[int, float, str] = val
        self._label: Label = label

    def value(self) -> Any:
        return self._event

    def label(self) -> Label:
        return self._label

    def max_representation(self) -> Any:
        raise NotImplementedError("BasicEvent.max_representation is not implemented")


class Memory:
    def __init__(self):
        self._events: List[MemoryEvent] = []

    def append(self, event: MemoryEvent):
        self._events.append(event)
