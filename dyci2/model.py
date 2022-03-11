# -*-coding:Utf-8 -*

#############################################################################
# model.py
# Models of symbolic sequences.
# Jérôme Nika, IRCAM STMS LAB / Ken Deguernel, INRIA Nancy - IRCAM STMS Lab
# copyleft 2016 - 2017
#############################################################################

# TODO : RAJOUTER DANS DOC LES ARGUMENTS LIES A "LABEL_TYPE" ET "CONTENT_TYPE"

"""
Model
===================

This module defines different models of symbolic sequences.
The classes defined in this module are minimal and only implement the construction algorithms and basic methods.
Navigation and creative aspects are handled by other classes in the library (cf. :mod:`Navigator` and
:mod:`ModelNavigator`).
Main classes: :class:`~Model.Model`, :class:`~Model.FactorOracle`.
Tutorial for the class :class:`~Model.FactorOracle` in :file:`_Tutorials_/FactorOracleAutomaton_tutorial.py`.

"""
import logging
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Type

from dyci2.dyci2_label import Dyci2Label
from merge.corpus import Corpus
from merge.main.candidate import Candidate
from merge.main.candidates import Candidates
from merge.main.corpus_event import CorpusEvent
from dyci2.parameter import Parametric
from dyci2.transforms import Transform


class Model(Parametric, ABC):
    """The class :class:`~Model.Model` is an **abstract class**.
    Any new model of sequence must inherit from this class.

    # :param sequence: sequence learnt in the model.
    # :type sequence: list or str
    # :param labels: sequence of labels chosen to describe the sequence.
    # :type labels: list or str
    # :param equiv: compararison function given as a lambda function, default if no parameter is given: self.equiv.
    # :type equiv: function

    # :!: **equiv** has to be consistent with the type of the elements in labels.
    # """

    # def __init__(self, memory: Corpus, equiv: Callable = (lambda x, y: x == y)):
    #     self.logger = logging.getLogger(__name__)
    #     self._memory: Corpus = memory
    #     self.equiv: Callable[[Dyci2Label, Dyci2Label], Dyci2Label] = equiv

    # TODO: Update model with correct type signatures
    # @abstractmethod
    # def learn_sequence(self, sequence: List[CorpusEvent], equiv: Optional[Callable] = None):
    #     """
    #     Learns (appends) a new sequence in the model.
    #
    #     :param sequence: sequence learnt in the Factor Oracle automaton
    #     :type sequence: list or str
    #     # :param labels: sequence of labels chosen to describe the sequence
    #     # :type labels: list or str
    #     :param equiv: Compararison function given as a lambda function, default if no parameter is given: self.equiv.
    #     :type equiv: function
    #
    #     :!: **equiv** has to be consistent with the type of the elements in labels.
    #
    #     """
    #
    # @abstractmethod
    # def learn_event(self, event: CorpusEvent, equiv: Optional[Callable] = None):
    #     """
    #     Learns (appends) a new state in the model.
    #
    #     :param event:
    #     # :param label:
    #     :param equiv: Compararison function given as a lambda function, default if no parameter is given: self.equiv.
    #     :type equiv: function
    #
    #     :!: **equiv** has to be consistent with the type of label.
    #
    #     """
    #
    # @abstractmethod
    # def select_events(self, index_state: int, label: Optional[Dyci2Label]) -> Candidates:
    #     """ TODO: Docstring """
    #
    # @abstractmethod
    # def feedback(self, output_event: Optional[Candidate]) -> None:
    #     """ TODO: Docstring """
    #
    # @abstractmethod
    # def encode_with_transform(self, transform: Transform) -> None:
    #     """ TODO: Docstring """
    #
    # @abstractmethod
    # def decode_with_transform(self, transform: Transform) -> None:
    #     """ TODO: Docstring """
