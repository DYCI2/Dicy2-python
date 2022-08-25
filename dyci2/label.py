# -*-coding:Utf-8 -*

#############################################################################
# label.py
# Axel Chemla--Romeu-Santos, IRCAM STMS LAB - Jérôme Nika, IRCAM STMS LAB 
# copyleft 2016 - 2017
#############################################################################

"""
Label
=========
Definition of alphabets of labels to build sequences and use them in creative applications.

"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Type, List, Any, Optional, Dict, Tuple

from merge.main.exceptions import LabelError
from merge.main.label import Label


class Dyci2Label(Label, ABC):

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self.label)})"

    def __str__(self):
        return str(self.label)

    @abstractmethod
    def __eq__(self, a):
        """ """

    @classmethod
    @abstractmethod
    def sequence_from_list(cls, init_list: List[str], **kwargs) -> List['Dyci2Label']:
        """ """

    @classmethod
    @abstractmethod
    def parse(cls, raw_data: Any) -> 'Dyci2Label':
        """ """

    @classmethod
    def type_from_string(cls, s: str) -> Type['Dyci2Label']:
        """ raises: TypeError if string doesn't match a type"""
        if s.lower() == "listlabel":
            return ListLabel
        elif s.lower() == "chordlabel":
            return ChordLabel
        else:
            raise LabelError(f"No label '{s}' exists")


class IntervallicLabel(Dyci2Label, ABC):
    """ Interface for Labels with intervallic/relative representations"""

    @classmethod
    @abstractmethod
    def equiv_mod_interval(cls, x: List[Any], y: List[Any]) -> bool:
        """ """

    @abstractmethod
    def delta(self, other: 'IntervallicLabel') -> Optional[int]:
        """ """

    @classmethod
    @abstractmethod
    def make_sequence_of_intervals_from_sequence_of_labels(cls, list_of_labels, **kwargs):
        """"""


class ChordLabel(IntervallicLabel):

    def __init__(self, label=None, first_chord_label=None, previous_chord_label=None):
        if type(label) == list and len(label) == 2:
            label = str(label[0]) + " " + str(label[1])

        Dyci2Label.__init__(self, label=label)

        if self.label is None:
            self.root = None
            self.chordtype = None
        else:
            self.root = self.get_root_from_label()
            self.normalize_root()
            self.chordtype = self.get_chordtype_from_label()
        self.interval_within_sequence: Dict[str, Optional[Tuple[Any, Any]]] = {}
        if first_chord_label is not None and issubclass(type(first_chord_label), ChordLabel):
            self.interval_within_sequence["first_chord_label"] = (first_chord_label.delta_root(self), first_chord_label)
        else:
            self.interval_within_sequence["first_chord_label"] = None
        if previous_chord_label is not None and issubclass(type(previous_chord_label), ChordLabel):
            self.interval_within_sequence["previous_chord_label"] = (previous_chord_label.delta_root(self),
                                                                     previous_chord_label)
        else:
            self.interval_within_sequence["previous_chord_label"] = None

    @classmethod
    def parse(cls, raw_data: Any) -> 'ChordLabel':
        if isinstance(raw_data, str):
            return cls.sequence_from_list([raw_data])[0]
        else:
            raise LabelError(f"class {cls.__name__} does not understand input of type {raw_data.__class__.__name__}")

    @classmethod
    def sequence_from_list(cls,
                           init_list: List[str],
                           init_first_chord_label=None,
                           init_previous_chord_label=None,
                           **kwargs) -> List['ChordLabel']:
        sequence = []
        if init_first_chord_label is not None:
            first_label = init_first_chord_label
        else:
            first_label = ChordLabel(init_list[0], first_chord_label=init_list[0], previous_chord_label=None)

        if init_previous_chord_label is not None:
            previous_label = init_previous_chord_label
        else:
            previous_label = None

        for item in init_list:
            new_label = ChordLabel(label=item, first_chord_label=first_label, previous_chord_label=previous_label)
            sequence.append(new_label)
            previous_label = new_label

        return sequence

    @classmethod
    def make_sequence_of_intervals_from_sequence_of_labels(cls, list_of_labels, *args, **kwargs):
        interval_sequence = []
        for label_str in list_of_labels:
            if label_str is not None:
                interval_sequence.append([label_str.delta_root_previous_label_in_sequence(), label_str.chordtype])
            else:
                interval_sequence.append([None, None])

        interval_sequence[0][0] = None

        return interval_sequence

    @classmethod
    def equiv_mod_interval(cls, x: List[Any], y: List[Any]) -> bool:
        return (x[1::] == y[1::]) and (x[0] == y[0] or x[0] is None or y[0] is None)

    @staticmethod
    def normalized_note(note) -> str:
        normalized_note = note
        if note == "db":
            normalized_note = "c#"
        elif note == "d#":
            normalized_note = "eb"
        elif note == "gb":
            normalized_note = "f#"
        elif note == "ab":
            normalized_note = "g#"
        elif note == "a#":
            normalized_note = "bb"
        elif note == "cb":
            normalized_note = "b"
        elif note == "b#":
            normalized_note = "c"
        elif note == "fb":
            normalized_note = "e"
        elif note == "e#":
            normalized_note = "f"
        return normalized_note

    def get_root_from_label(self) -> str:
        return self.label.split(" ")[0].lower()

    def get_chordtype_from_label(self) -> str:
        return self.label.split(" ")[1].lower()

    def normalize_root(self) -> None:
        self.root = self.normalized_note(self.root)

    def __eq__(self, a):
        if a is None:
            return False
        elif isinstance(a, ChordLabel):
            return self.normalized_note(self.root) == self.normalized_note(a.root) and self.chordtype == a.chordtype
        elif isinstance(a, list) or isinstance(a, tuple):
            return self.root == self.normalized_note(a[0]) and self.chordtype == a[1]
        else:
            raise LabelError(f"Failed comparing chord label with {a.__repr__()}")

    def transpose_root(self, i: int) -> None:
        self.normalize_root()
        key_labels = ["c", "c#", "d", "eb", "e", "f", "f#", "g", "g#", "a", "bb", "b",
                      "c", "c#", "d", "eb", "e", "f", "f#", "g", "g#", "a", "bb", "b"]
        pos = key_labels.index(self.root)
        key_labels = key_labels[pos::]
        self.root = key_labels[i % 12]
        self.label = self.root + " " + self.chordtype

    def delta_root(self, c: 'ChordLabel') -> Optional[int]:
        if self is None or c is None or self.label is None or c.label is None:
            return None
        key_labels = ["c", "c#", "d", "eb", "e", "f", "f#", "g", "g#", "a", "bb", "b"]
        p1 = key_labels.index(self.normalized_note(self.root))
        p2 = key_labels.index(self.normalized_note(c.root))
        return ((p2 - p1 + 5) % 12) - 5

    def delta(self, a) -> Optional[int]:
        if self.chordtype != a.chordtype:
            return None
        else:
            return self.delta_root(a)

    def delta_root_first_label_in_sequence(self):
        if self is None:
            return None
        delta_first_label = self.interval_within_sequence.get("first_chord_label")
        if delta_first_label and delta_first_label is not None:
            return delta_first_label[0]
        else:
            return None

    def delta_root_previous_label_in_sequence(self):
        if self is None:
            return None
        delta_previous_label = self.interval_within_sequence.get("previous_chord_label")
        if delta_previous_label and delta_previous_label is not None:
            return delta_previous_label[0]
        else:
            return None


class ListLabel(Dyci2Label):

    def __init__(self, label: Optional[List[Any]] = None, depth: Optional[int] = None):
        super().__init__(label=label if label is not None else [None])
        self.depth: int = len(self.label) if depth is None else min(depth, len(self.label))

    def __repr__(self):
        return f"{self.__class__.__name__}(label={self.label},depth={self.depth}"

    def __str__(self):
        s = "List label: " + str(self.label)
        if self.depth:
            s += f"(depth = {self.depth})"
        return s

    def __eq__(self, a: 'Dyci2Label'):
        if a is None:
            return False
        elif isinstance(a, self.__class__):
            result: bool = True
            i: int = 0
            depth: int = max(1, min(self.depth, a.depth, len(self.label), len(a.label)))
            while result and i < depth:
                if not (self.label[i] is None or a.label[i] is None or self.label[i] == a.label[i]):
                    result = False
                i += 1
            return result
        elif isinstance(a, list) or isinstance(a, tuple):
            result = True
            i = 0
            depth = max(1, min(self.depth, len(self.label), len(a)))
            while result and i < depth:
                if not (self.label[i] is None or a[i] is None or self.label[i] == a[i]):
                    result = False
                i += 1
            return result
        else:
            raise LabelError(f"Failed comparing {self.__class__.__name__} with label of type {a.__class__.__name__}")

    @classmethod
    def parse(cls, raw_data: Any) -> 'ListLabel':
        if isinstance(raw_data, str):
            return cls(raw_data.split(" "))
        elif isinstance(raw_data, Iterable):
            return cls(list(raw_data))
        else:
            return cls([raw_data])

    @classmethod
    def sequence_from_list(cls, init_list: List[str], **kwargs) -> List['ListLabel']:
        return [ListLabel([e]) for e in init_list]
