from typing import Optional

from memory import MemoryEvent


class Candidate:
    def __init__(self, event: MemoryEvent, index: int, score: float, transform: Optional['Transform'] = None):
        self.event: MemoryEvent = event
        self.index: int = index
        self.score: float = score
        self.transform: Optional['Transform'] = transform

    def __repr__(self):
        return f"{self.__class__.__name__}(event={self.event},index={self.index}," \
               f"score={self.score},transform={self.transform})"
