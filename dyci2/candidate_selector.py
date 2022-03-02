import random
import warnings
from abc import ABC
from typing import List, Optional

from merge.main.candidate import Candidate
from merge.main.candidates import Candidates
from merge.main.jury import Jury
from parameter import Parametric


class CandidateSelector(Jury, Parametric, ABC):
    pass


class TempCandidateSelector(CandidateSelector):
    def decide(self, candidates: Candidates) -> Optional[Candidate]:
        if candidates.size() == 0:
            return None
        else:
            return candidates.get_candidate(0)

    def feedback(self, candidate: Optional[Candidate], **kwargs) -> None:
        pass

    def clear(self) -> None:
        pass


class DefaultFallbackSelector(CandidateSelector):
    def __init__(self):
        # TODO: Need to handle execution/generation trace properly here
        self.previous_output: Optional[Candidate] = None

    def decide(self, candidates: Candidates) -> Optional[Candidate]:
        print("NO EMPTY EVENT")
        all_memory: List[Candidate] = candidates.get_candidates()
        if self.previous_output is not None:
            warnings.warn("This will probably cause issues if the FOModel's initial None still is a part of the Memory")
            next_index: int = self.previous_output.event.index + 1
            if next_index < len(all_memory):
                return all_memory[next_index]

        if len(all_memory) > 0:
            next_index: int = random.randint(0, len(all_memory) - 1)
            return all_memory[next_index]

        return None

    def feedback(self, candidate: Optional[Candidate], **kwargs) -> None:
        self.previous_output = candidate

    def clear(self) -> None:
        self.previous_output = None
