import copy
import logging
from typing import List, Optional, Tuple

from dyci2.corpus_event import Dyci2CorpusEvent
from merge.main.candidate import Candidate
from merge.main.exceptions import StateError


class GenerationHistory:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._generation_trace: List[Optional[Candidate]] = []
        self._generation_time: int = -1
        self._last_sequence: Optional[Tuple[int, int]] = None

    def clear(self) -> None:
        self._generation_trace: List[Optional[Candidate]] = []
        self._generation_time: int = -1
        self._last_sequence = None

    def add_output(self, generation_index: int, generation_output: List[Optional[Candidate]]):
        generated_output_length: int = len(generation_output)

        prev_generation_time: int = self._generation_time

        if generation_index > len(self._generation_trace):
            for _ in range(len(self._generation_trace), generation_index):
                self._generation_trace.append(None)

        for i in range(generated_output_length):
            if i + generation_index < len(self.generation_trace):
                self.generation_trace[generation_index + i] = generation_output[i]
            else:
                self.generation_trace.append(generation_output[i])

        self._generation_time = generation_index + generated_output_length
        self._last_sequence = generation_index, generation_index + len(generation_output)
        self.logger.debug(f"generation time: {prev_generation_time} --> {self._generation_time}")

    def last_sequence(self) -> List[Optional[Candidate]]:
        """ raises: StateError if no sequence has been generated """
        try:
            return self.generation_trace[self._last_sequence[0]:self._last_sequence[1]]
        except (TypeError, IndexError):
            raise StateError("Generation trace is empty")

    def start_index_of_last_sequence(self) -> int:
        return self._last_sequence[0]

    def update_generation_time(self, new_time: int):
        self._generation_time = new_time

    @property
    def generation_time(self) -> int:
        return self._generation_time

    @property
    def generation_trace(self) -> List[Optional[Candidate]]:
        return self._generation_trace

    # def formatted_output_couple_content_transfo(self):
    #     """ raises: StateError if GenerationProcess has an event of an invalid type """
    #     output: List[Tuple[str, int]] = []
    #     for candidate in self.last_sequence():
    #         if candidate is not None:
    #             # TODO: Migrate this behaviour to Renderable interface
    #             if isinstance(candidate.event, Dyci2CorpusEvent):
    #                 output.append((candidate.event.renderer_info(), candidate.transform.renderer_info()))
    #             else:
    #                 raise StateError(f"Invalid event of type {type(candidate.event)} encountered")
    #         else:
    #             output.append(("None", 0))
    #     return output
