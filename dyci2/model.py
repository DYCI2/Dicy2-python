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
from typing import List, Optional, TypeVar, Generic

from dyci2.equiv import Equiv
from dyci2.label import Dyci2Label
from dyci2.parameter import Parametric
from dyci2.transforms import Transform
from merge.main.candidate import Candidate

T = TypeVar('T')


class Model(Parametric, Generic[T], ABC):
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

    @abstractmethod
    def learn_sequence(self,
                       sequence: List[Optional[T]],
                       labels: List[Optional[Dyci2Label]],
                       equiv: Optional[Equiv] = None) -> None:
        """ TODO: Update Docstring
        Learns (appends) a new sequence in the model.

        #:param sequence: sequence learnt in the Factor Oracle automaton
        #:type sequence: list or str
        # :param labels: sequence of labels chosen to describe the sequence
        # :type labels: list or str
        #:param equiv: Compararison function given as a lambda function, default if no parameter is given: self.equiv.
        #:type equiv: function

        #:!: **equiv** has to be consistent with the type of the elements in labels.
        """

    @abstractmethod
    def learn_event(self,
                    event: Optional[T],
                    label: Optional[Dyci2Label],
                    equiv: Optional[Equiv] = None) -> None:
        """ TODO: Docstring (can be copied/moved from FactorOracle, probably)
        Learns (appends) a new state in the model.

        # :param event:
        # :param label:
        # :param equiv: Compararison function given as a lambda function, default if no parameter is given: self.equiv.
        # :type equiv: function
        #
        # :!: **equiv** has to be consistent with the type of label.

        """

    @abstractmethod
    def feedback(self, output_event: Optional[Candidate]) -> None:
        """ TODO: Docstring """

    @abstractmethod
    def encode_with_transform(self, transform: Transform) -> None:
        """ TODO: Docstring """

    @abstractmethod
    def decode_with_transform(self, transform: Transform) -> None:
        """ TODO: Docstring """

    @abstractmethod
    def clear(self) -> None:
        """ TODO: Docstring """
