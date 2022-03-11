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
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Callable

from dyci2.dyci2_label import Dyci2Label
from merge.corpus import Corpus
from merge.main.candidate import Candidate
from merge.main.candidates import Candidates
from merge.main.corpus_event import CorpusEvent
from dyci2.parameter import Parametric


class Navigator(Parametric, ABC):
    """ """

    # def __init__(self, memory: Corpus, equiv: Callable = (lambda x, y: x == y), **kwargs):
    #     self.logger = logging.getLogger(__name__)
    #     self.memory: Corpus = memory
    #     self.equiv: Callable = equiv

    # TODO: Update signatures
    # @abstractmethod
    # def learn_sequence(self, sequence: List[CorpusEvent], equiv: Optional[Callable] = None):
    #     """ TODO: Docstring """
    #
    # @abstractmethod
    # def learn_event(self, event: CorpusEvent, equiv: Optional[Callable] = None):
    #     """ TODO: Docstring """
    #
    # @abstractmethod
    # def rewind_generation(self, index_state: int):
    #     """ TODO: Docstring """
    #
    # @abstractmethod
    # def weight_candidates(self, candidates: Candidates, required_label: Optional[Dyci2Label], **kwargs) -> Candidates:
    #     """ TODO: Docstring """
    #
    # @abstractmethod
    # def clear(self):
    #     """ TODO: Docstring """
    #
    # @abstractmethod
    # def feedback(self, output_event: Optional[Candidate]) -> None:
    #     """ TODO: Docstring """
