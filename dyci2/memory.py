import copy
from abc import ABC, abstractmethod
from typing import Any, List, Union, Type, TypeVar, Generic

from label import Label

T = TypeVar('T')


class MemoryEvent(ABC, Generic[T]):
    def __init__(self):
        pass

    @abstractmethod
    def data(self) -> T:
        """ TODO """

    @abstractmethod
    def label(self) -> Label:
        """ TODO """

    @abstractmethod
    def renderer_info(self) -> str:
        """ TODO """


class BasicEvent(MemoryEvent[Union[int, float, str]]):
    def __init__(self, data: Union[int, float, str], label: Label):
        super().__init__()
        self._data: Union[int, float, str] = data
        self._label: Label = label

    def data(self) -> Union[int, float, str]:
        return self._data

    def label(self) -> Label:
        return self._label

    def renderer_info(self) -> str:
        return str(self._data)


class DebugEvent(MemoryEvent[Label]):
    def __init__(self, data: Label, label: Label):
        super().__init__()
        self._data: Label = data
        self._label: Label = copy.deepcopy(label)

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data()},label={self.label()})"

    def data(self) -> Label:
        return self._data

    def label(self) -> Label:
        return self._label

    def renderer_info(self) -> str:
        raise NotImplementedError("DebugEvent.max_representation is not implemented")


class Memory:
    def __init__(self, events: List[MemoryEvent], content_type: Type[MemoryEvent], label_type: Type[Label]):
        self.events: List[MemoryEvent] = events
        self.content_type: Type[MemoryEvent] = content_type
        self.label_type: Type[Label] = label_type

    @classmethod
    def new_empty(cls, content_type: Type[MemoryEvent], label_type: Type[Label]):
        return cls([], content_type, label_type)

    def append(self, event: MemoryEvent):
        self.events.append(event)

    def length(self) -> int:
        return len(self.events)
