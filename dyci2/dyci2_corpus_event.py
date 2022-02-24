from typing import TypeVar, Any

from dyci2_label import Dyci2Label
from merge.main.corpus_event import CorpusEvent

T = TypeVar('T')


class Dyci2CorpusEvent(CorpusEvent):
    def __init__(self, data: Any, label: Dyci2Label, index: int, *args, **kwargs):
        super().__init__(index, {}, {type(label): label}, *args, **kwargs)
        self.data: Any = data

    def renderer_info(self) -> str:
        return str(self.data)


# class MemoryEvent(ABC, Generic[T]):
#     def __init__(self):
#         pass
#
#     @abstractmethod
#     def __str__(self):
#         """ TODO: Docstring """
#
#     @abstractmethod
#     def data(self) -> T:
#         """ TODO: Docstring """
#
#     @abstractmethod
#     def label(self) -> Label:
#         """ TODO: Docstring """
#
#     @abstractmethod
#     def renderer_info(self) -> str:
#         """ TODO: Docstring """


# class BasicEvent(MemoryEvent[Union[int, float, str]]):
#     def __init__(self, data: Union[int, float, str], label: Label):
#         super().__init__()
#         self._data: Union[int, float, str] = data
#         self._label: Label = label
#
#     def __str__(self):
#         return f"({self._data},{self._label})"
#
#     def data(self) -> Union[int, float, str]:
#         return self._data
#
#     def label(self) -> Label:
#         return self._label
#
#     def renderer_info(self) -> str:
#         return str(self._data)
#
#
# class LabelEvent(MemoryEvent[Label]):
#     def __init__(self, data: Label, label: Label):
#         super().__init__()
#         self._data: Label = data
#         self._label: Label = copy.deepcopy(label)
#
#     def __str__(self):
#         return f"({self._data},{self._label})"
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}(data={self.data()},label={self.label()})"
#
#     def data(self) -> Label:
#         return self._data
#
#     def label(self) -> Label:
#         return self._label
#
#     def renderer_info(self) -> str:
#         raise NotImplementedError("DebugEvent.max_representation is not implemented")


