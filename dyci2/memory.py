from abc import ABC, abstractmethod
from typing import Any, List, Union, Type

from label import Label


class MemoryEvent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def event(self) -> Any:
        """ TODO """

    @abstractmethod
    def label(self) -> Label:
        """ TODO """

    @abstractmethod
    def max_representation(self) -> Any:
        """ TODO """


class BasicEvent(MemoryEvent):
    def __init__(self, event: Union[int, float, str], label: Label):
        super().__init__()
        self._event: Union[int, float, str] = event
        self._label: Label = label

    def event(self) -> Any:
        return self._event

    def label(self) -> Label:
        return self._label

    def max_representation(self) -> Any:
        raise NotImplementedError("BasicEvent.max_representation is not implemented")


class DebugEvent(MemoryEvent):
    def __init__(self, event: Label, label: Label):
        super().__init__()
        self._event: Label = event
        self._label: Label = label

    def event(self) -> Label:
        return self._event

    def label(self) -> Label:
        return self._label

    def max_representation(self) -> Any:
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
