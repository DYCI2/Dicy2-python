import random
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional

from candidate import Candidate
from candidates import Candidates
from parameter import Parametric


class CandidateSelector(Parametric, ABC):
    @abstractmethod
    def decide(self, candidates: Candidates) -> Optional[Candidate]:
        """ TODO """

    @abstractmethod
    def feedback(self, time: int, output_event: Optional[Candidate]) -> None:
        """ TODO """


class TempCandidateSelector(CandidateSelector):
    def decide(self, candidates: Candidates) -> Optional[Candidate]:
        if candidates.length() == 0:
            return None
        else:
            return candidates.at(0)

    def feedback(self, time: int, output_event: Optional[Candidate]) -> None:
        pass


class DefaultFallbackSelector(CandidateSelector):
    def __init__(self):
        self.previous_output: Optional[
            Candidate] = None  # TODO: Need to handle execution/generation trace properly here

    def decide(self, candidates: Candidates) -> Optional[Candidate]:
        print("NO EMPTY EVENT")
        all_memory: List[Candidate] = candidates.memory_as_candidates()
        if self.previous_output is not None:
            warnings.warn("This will probably cause issues if the FOModel's initial None still is a part of the Memory")
            next_index: int = self.previous_output.index + 1
            if next_index < len(all_memory):
                return all_memory[next_index]

        if len(all_memory) > 0:
            next_index: int = random.randint(0, len(all_memory) - 1)
            return all_memory[next_index]

        return None

    def feedback(self, time: int, output_event: Optional[Candidate]) -> None:
        self.previous_output = output_event
