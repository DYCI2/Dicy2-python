# -*-coding:Utf-8 -*

####################################################################################
# navigator.py
# Parameters and methods to navigate through a symbolic sequence.
# Jérôme Nika, Ken Déguernel - IRCAM STMS Lab
# copyleft 2016 - 2018
####################################################################################

# TODO
# ICI ET DANS LES MODEL NAVIGATORS : TESTER SI LE REQUIRED LABEL EST BIEN DU MEME TYPE QUE LES LABELS DANS LE MODEL

"""
Navigator
===================

This module defines parameters and methods to navigate through a symbolic sequence.
The classes defined in this module are used in association with models (cf. :mod:`Model`) when creating
**model navigator** classes (cf. :mod:`ModelNavigator`).

"""
from abc import ABC, abstractmethod
from typing import List, Optional, Callable

from candidate import Candidate
from candidates import Candidates
from label import Label
from memory import MemoryEvent, Memory
from parameter import Parametric


class Navigator(Parametric, ABC):
    def __init__(self, memory: Memory, equiv: Callable = (lambda x, y: x == y), **kwargs):
        self.memory: Memory = memory
        self.equiv: Callable = equiv

    @abstractmethod
    def learn_sequence(self, sequence: List[MemoryEvent], equiv: Optional[Callable] = None):
        """ TODO: Docstring """

    @abstractmethod
    def learn_event(self, event: MemoryEvent, equiv: Optional[Callable] = None):
        """ TODO: Docstring """

    @abstractmethod
    def rewind_generation(self, index_state: int):
        """ TODO: Docstring """

    @abstractmethod
    def weight_candidates(self, candidates: Candidates, required_label: Optional[Label], **kwargs) -> Candidates:
        """ TODO: Docstring """

    @abstractmethod
    def clear(self):
        """ TODO: Docstring """

    @abstractmethod
    def feedback(self, output_event: Optional[Candidate]) -> None:
        """ TODO: Docstring """
