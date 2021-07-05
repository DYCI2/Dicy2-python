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
from abc import ABC, abstractmethod
from typing import Callable

from candidate import Candidate
from candidates import Candidates
from dyci2.label import *
from memory import MemoryEvent, Memory
from parameter import Parametric


class Model(Parametric, ABC):
    
    """The class :class:`~Model.Model` is an **abstract class**.
    Any new model of sequence must inherit from this class.

    # :param sequence: sequence learnt in the model.
    # :type sequence: list or str
    # :param labels: sequence of labels chosen to describe the sequence.
    # :type labels: list or str
    :param equiv: compararison function given as a lambda function, default if no parameter is given: self.equiv.
    :type equiv: function

    :!: **equiv** has to be consistent with the type of the elements in labels.
    """

    def __init__(self, memory: Memory, equiv: Callable = (lambda x, y: x == y)):
        """ TODO[B]: Note that memory must always exist (as it may be necessary to call some sort of `build` or
             `init_memory` from constructor) but it may be empty. """
        
        self._memory: Memory = memory
        self.equiv: Callable[[Label, Label], Label] = equiv

    @abstractmethod
    def learn_sequence(self, sequence: List[MemoryEvent], equiv: Optional[Callable] = None):
        
        """
        Learns (appends) a new sequence in the model.

        :param sequence: sequence learnt in the Factor Oracle automaton
        :type sequence: list or str
        # :param labels: sequence of labels chosen to describe the sequence
        # :type labels: list or str
        :param equiv: Compararison function given as a lambda function, default if no parameter is given: self.equiv.
        :type equiv: function

        :!: **equiv** has to be consistent with the type of the elements in labels.

        """

    @abstractmethod
    def learn_event(self, event: MemoryEvent, equiv: Optional[Callable] = None):
        
        """
        Learns (appends) a new state in the model.

        :param event:
        # :param label:
        :param equiv: Compararison function given as a lambda function, default if no parameter is given: self.equiv.
        :type equiv: function

        :!: **equiv** has to be consistent with the type of label.

        """

    @abstractmethod
    def get_candidates(self, index_state: int, label: Optional[Label]) -> Candidates:
        
        """ TODO """

    @abstractmethod
    @property
    def memory(self):
        """ TODO """

    @abstractmethod
    def print_model(self):
        
        # TODO: Should this really be an enforced method?
        """ TODO """

    @abstractmethod
    def memory_length(self):
        
        """ TODO """

    @abstractmethod
    def feedback(self, time: int, output_event: Optional[Candidate]) -> None:
        """ TODO """

