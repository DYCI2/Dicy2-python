import warnings
from typing import List

# TODO: Abstract base class, so that this can be extended with numpy peaks, etc.
# class AbstractCandidates(ABC):
#
# class OptimizedCandidates(AbstractCandidates):
#     def __init__(self):
#         self.scores = np.array()
#         self.indices = np.array()
#         self.transforms = np.array()
#         self.memory = memory
#
from candidate import Candidate
from memory import Memory


class Candidates:
    def __init__(self, candidates: List[Candidate], memory: Memory):
        self.data: List[Candidate] = candidates
        self.memory: Memory = memory
        warnings.warn("the Candidates class does not handle transforms")


    def append(self, candidate: Candidate):
        self.data.append(candidate)

    def length(self) -> int:
        return len(self.data)

    def at(self, index: int) -> Candidate:
        """ raises IndexError if index is out of range """
        return self.data[index]

    # TODO: (Ideally) Remove Optional once properly handled in FactorOracleNavigator
    def memory_as_candidates(self, exclude_last: bool = False,
                             exclude_first: bool = True) -> List[Candidate]:
        start: int = 1 if exclude_first else 0
        end: int = self.memory.length() - 1 if exclude_last else self.memory.length()
        return [Candidate(self.memory.events[i], i, 1.0, None) for i in range(start, end)]
