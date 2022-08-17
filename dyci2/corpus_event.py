from typing import TypeVar

from merge.main.corpus_event import GenericCorpusEvent
from merge.main.label import Label

T = TypeVar('T')


class Dyci2CorpusEvent(GenericCorpusEvent[T]):
    def __init__(self, data: T, index: int, label: Label):
        super().__init__(data=data, index=index, descriptors=None, labels={type(label): label})

    def __str__(self):
        return f"{self.__class__.__name__}({str(self.data)})"

    def renderer_info(self) -> str:
        return str(self.data)
