from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple

from dyci2.corpus_event import Dyci2CorpusEvent
from merge.main.candidate import Candidate
from merge.main.exceptions import StateError


class FormattingUtils:
    @staticmethod
    def output_without_transforms(candidates: List[Optional[Candidate]], use_max_format: bool = True) -> str:
        if use_max_format:
            return " ".join(e_str for e_str, _ in FormattingUtils._format_output(candidates))
        else:
            return str([e_str for e_str, _ in FormattingUtils._format_output(candidates)])

    @staticmethod
    def _format_output(candidates: List[Optional[Candidate]]) -> List[Tuple[str, int]]:
        output: List[Tuple[str, int]] = []
        for candidate in candidates:
            if candidate is not None:
                # TODO: Migrate this behaviour to Renderable interface
                if isinstance(candidate.event, Dyci2CorpusEvent):
                    output.append((candidate.event.renderer_info(), candidate.transform.renderer_info()))
                else:
                    raise StateError(f"Invalid event of type {type(candidate)} encountered")
            else:  # candidate is None
                output.append((str(candidate), 0))

        return output


# TODO: Remove
def format_list_as_list_of_strings(l):
    result = []
    for i in l:
        i_s = ""
        if type(i) == list:
            for j in range(len(i) - 1):
                i_s += format_obj_as_string(i[j]) + " "
            i_s += format_obj_as_string(i[len(i) - 1])
        else:
            i_s = format_obj_as_string(i)
        result.append(i_s)

    return result


# TODO: Remove
def format_obj_as_string(o):
    s = ""
    if type(o) == str:
        s = o.replace("u'", "").replace("'", "")
    else:
        s = format(o)
    return s


def none_is_infinite(value):
    if value is None:
        return float("inf")
    else:
        return value
