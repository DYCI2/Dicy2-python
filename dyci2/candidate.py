from typing import Optional

from memory import MemoryEvent


class Candidate:
    def __init__(self, event: MemoryEvent, index: int, score: float, transform: Optional[int] = None):
        self.event: MemoryEvent = event
        self.index: int = index
        self.score: float = score
        # TODO: Should be Transform (or even List[Transform]), not int, once properly implemented
        self.transform: Optional[int] = transform
