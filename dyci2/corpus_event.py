from typing import TypeVar

from merge.main.corpus_event import GenericCorpusEvent

T = TypeVar('T')


class Dyci2CorpusEvent(GenericCorpusEvent):
    def renderer_info(self) -> str:
        return str(self.data)
