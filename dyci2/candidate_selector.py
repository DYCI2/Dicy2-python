from abc import ABC, abstractmethod
from typing import List, Optional

from candidate import Candidate


class CandidateSelector(ABC):
    @abstractmethod
    def decide(self, candidates: List[Candidate]) -> Optional[Candidate]:
        """ TODO """


class TempCandidateSelector(CandidateSelector):
    def decide(self, candidates: List[Candidate]) -> Optional[Candidate]:
        if len(candidates) == 0:
            return None
        else:
            return candidates[0]
