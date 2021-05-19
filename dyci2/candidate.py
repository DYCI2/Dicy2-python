from memory import MemoryEvent


class Candidate:
    def __init__(self, event: MemoryEvent, index: int, score: float, transform: int):
        self.event: MemoryEvent = event
        self.index: int = index
        self.score: float = score
        # TODO: Should be Transform (or even List[Transform]), not int, once properly implemented
        self.transform: int = transform
