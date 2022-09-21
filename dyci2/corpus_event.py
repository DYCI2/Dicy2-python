from collections.abc import Iterable
from typing import TypeVar, Any

from merge.main.corpus_event import GenericCorpusEvent
from merge.main.label import Label

T = TypeVar('T')


class Dyci2CorpusEvent(GenericCorpusEvent[T]):
    """ Extension of the :class:`~merge.GenericCorpusEvent` intended as the memory of the :class:`~merge.Corpus`,
    with the addition of a function for rendering its data over OSC """
    def __init__(self, data: T, index: int, label: Label):
        super().__init__(data=data, index=index, descriptors=None, labels={type(label): label})

    def __str__(self):
        return f"{self.__class__.__name__}({str(self.data)})"

    def renderer_info(self) -> Any:
        if isinstance(self.data, str):  # Only needed to make sure that string isn't treated as Iterable
            return self.data
        elif isinstance(self.data, Iterable):
            return " ".join([str(e) for e in self.data])
        else:
            return self.data
