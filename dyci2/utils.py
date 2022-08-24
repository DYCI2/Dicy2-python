from typing import List, Optional, Tuple, Any

from dyci2.corpus_event import Dyci2CorpusEvent
from merge.main.candidate import Candidate
from merge.main.exceptions import StateError, QueryError


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


class GenerationTraceFormatter:
    @staticmethod
    def query(keyword: str,
              generation_trace: List[Optional[Candidate]],
              start: Optional[int] = None,
              end: Optional[int] = None) -> List[Any]:
        if not keyword or keyword == "bang":
            return [FormattingUtils.output_without_transforms(generation_trace, use_max_format=True)]

        elif keyword.lower() == "len" or keyword.lower() == "length":
            return [keyword, len(generation_trace)]

        elif keyword.lower() == "range":
            if start is not None and not isinstance(start, int):
                raise QueryError(f"Invalid start index ({start}) for keyword '{keyword}': expected integer or 'None'")
            if end is not None and not isinstance(start, int):
                raise QueryError(f"Invalid end index ({start}) for keyword '{keyword}': expected integer or 'None'")
            return [keyword, FormattingUtils.output_without_transforms(generation_trace[start:end])]

        elif keyword.lower() == "mth":
            if start is None:
                raise QueryError(f"Missing argument for keyword '{keyword}'.")
            elif not isinstance(start, int):
                raise QueryError(f"Invalid argument for keyword '{keyword}': expected integer (actual: {type(start)}")
            else:
                try:
                    return [FormattingUtils.output_without_transforms([generation_trace[start]])]
                except IndexError:
                    raise QueryError(f"Index {start} is out of range (valid range is 0, {len(generation_trace) - 1})")
        else:
            raise QueryError(f"Invalid keyword '{keyword}'. Valid keywords are: 'len', 'range', 'mth'")


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
