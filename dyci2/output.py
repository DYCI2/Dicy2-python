import copy
from typing import Optional

from candidate import Candidate
from memory import MemoryEvent
from transforms import Transform, NoTransform


class Output:
    def __init__(self, event: MemoryEvent, index: int, applied_transform: Transform):
        self.event: MemoryEvent = event
        self.index: int = index
        self.applied_transform: Transform = applied_transform

    # TODO: The once transforms are correctly handled, remove this function and just do deepcopying and transforming
    #  directly in __init__. But pass event + transform_to_apply, not just candidate (too implicit)
    @classmethod
    def from_candidate(cls, candidate: Candidate,
                       transform_to_apply: Optional[Transform]) -> 'Output':
        # TODO: Handle transforms properly. Candidate should already contain a transform and thus not need this.
        if transform_to_apply is not None:
            transform: Transform = transform_to_apply
        else:
            transform = NoTransform()

        event_cloned: MemoryEvent = copy.deepcopy(candidate.event)
        event: MemoryEvent = transform.encode(event_cloned)

        return cls(event=event, index=candidate.index, applied_transform=transform)
