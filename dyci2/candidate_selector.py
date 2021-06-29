from abc import ABC, abstractmethod
from typing import List, Optional

from candidate import Candidate
from candidates import Candidates


class CandidateSelector(ABC):
    @abstractmethod
    def decide(self, candidates: Candidates) -> Optional[Candidate]:
        """ TODO """


class TempCandidateSelector(CandidateSelector):
    def decide(self, candidates: Candidates) -> Optional[Candidate]:
        if candidates.length() == 0:
            return None
        else:
            return candidates.at(0)
